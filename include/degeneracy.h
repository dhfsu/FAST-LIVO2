/*
Degeneracy detection utilities for FAST-LIVO2.

Self-contained, pure-Eigen (no ROS / PCL dependency) so it can be unit-tested in
isolation. It analyses the 6x6 pose information matrix (H^T R^-1 H) that both the
LIO (voxel_map.cpp) and VIO (vio.cpp) IESKF updates already compute, and reports
per-direction degeneracy.

Design notes:
  - The 6x6 information block mixes rotation (cols 0-2) and translation (cols 3-5)
    which carry different physical units, so the two 3x3 sub-blocks are analysed
    SEPARATELY. This keeps the reported eigen-directions physically interpretable
    (a translation eigenvector is a world-frame axis) and lets each block use its
    own threshold.
  - A direction is flagged degenerate if EITHER the block condition number exceeds
    cond_thresh (relative, scale-invariant, catches corridors/tunnels) OR the
    smallest eigenvalue drops below an absolute floor (catches globally weak
    observability that the condition number alone would miss).
  - For LiDAR an additional, well-normalised cross-check is available: the scatter
    matrix of matched plane normals. Its eigenvalues sum to 1 (unit normals), so a
    smallest eigenvalue near zero directly means "no plane constrains this axis".
  - Temporal hysteresis latches the boolean flag to avoid per-frame flicker.

This module performs DETECTION ONLY. It never modifies the state or covariance.
*/

#ifndef DEGENERACY_H_
#define DEGENERACY_H_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <vector>

namespace degeneracy
{

struct DegeneracyConfig
{
  bool enable = false;             // master switch; when false no analysis runs
  double cond_thresh = 100.0;      // condition number (lambda_max/lambda_min) above which a block is degenerate
  double lambda_floor_rot = 1e-3;  // floor on the STRONGEST rotation-info eigenvalue; below it the block is uninformative
  double lambda_floor_trans = 1e-3;// floor on the STRONGEST translation-info eigenvalue; below it the block is uninformative
  double normal_scatter_thresh = 0.02; // LiDAR only: min eigenvalue of the (normalised) plane-normal scatter matrix
  int hysteresis_on = 3;           // consecutive raw-degenerate frames required to latch ON
  int hysteresis_off = 3;          // consecutive good frames required to latch OFF
};

struct DegeneracyResult
{
  // Per-block eigen-analysis (ascending eigenvalues; index 0 is the weakest direction).
  Eigen::Vector3d eval_rot = Eigen::Vector3d::Zero();
  Eigen::Vector3d eval_trans = Eigen::Vector3d::Zero();
  double cond_rot = 1.0;
  double cond_trans = 1.0;
  bool rot_degenerate = false;
  bool trans_degenerate = false;
  Eigen::Vector3d rot_degenerate_dir = Eigen::Vector3d::Zero();   // weakest rotation axis (state tangent space)
  Eigen::Vector3d trans_degenerate_dir = Eigen::Vector3d::Zero(); // weakest translation axis (world frame)

  // Auxiliary metrics.
  double normal_scatter_min = -1.0; // LiDAR plane-normal scatter min eigenvalue; -1 when not computed
  int effective_num = 0;            // number of observations used in the update

  // Combined verdict.
  bool degenerate_raw = false; // instantaneous (this frame)
  bool degenerate = false;     // latched through hysteresis
  double soft_factor = 1.0;    // continuous confidence: 0 = fully degenerate, 1 = well conditioned
};

// Analyse the 6x6 pose information block. Rows/cols 0-2 are rotation, 3-5 translation.
inline DegeneracyResult analyzePoseInformation(const Eigen::Matrix<double, 6, 6> &HTH, const DegeneracyConfig &cfg)
{
  DegeneracyResult r;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot(HTH.block<3, 3>(0, 0));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_trans(HTH.block<3, 3>(3, 3));
  r.eval_rot = es_rot.eigenvalues();     // ascending
  r.eval_trans = es_trans.eigenvalues(); // ascending
  r.rot_degenerate_dir = es_rot.eigenvectors().col(0);
  r.trans_degenerate_dir = es_trans.eigenvectors().col(0);

  const double eps = 1e-12;
  double lr_min = std::max(r.eval_rot(0), 0.0);
  double lr_max = std::max(r.eval_rot(2), 0.0);
  double lt_min = std::max(r.eval_trans(0), 0.0);
  double lt_max = std::max(r.eval_trans(2), 0.0);
  r.cond_rot = lr_max / std::max(lr_min, eps);
  r.cond_trans = lt_max / std::max(lt_min, eps);

  // A block is degenerate if it carries essentially NO information (its strongest
  // eigenvalue is below the floor) OR it is strongly anisotropic (condition number
  // above threshold, i.e. some direction is far weaker than the best-observed one).
  // Testing lambda_max (not lambda_min) for the "no information" case is robust: the
  // smallest eigenvalue is numerically wild (can be ~0 whenever any single direction
  // is unconstrained), whereas the largest is stable.
  bool rot_has_info = (lr_max >= cfg.lambda_floor_rot);
  bool trans_has_info = (lt_max >= cfg.lambda_floor_trans);
  r.rot_degenerate = !rot_has_info || (r.cond_rot > cfg.cond_thresh);
  r.trans_degenerate = !trans_has_info || (r.cond_trans > cfg.cond_thresh);

  // Continuous confidence in [0,1]: 0 when a block is uninformative or at/over the
  // condition-number threshold, 1 when both blocks are perfectly isotropic. Log-scaled
  // on the worst-block condition number so it degrades smoothly as the scene stiffens.
  if (!rot_has_info || !trans_has_info)
  {
    r.soft_factor = 0.0;
  }
  else
  {
    double denom = std::log10(std::max(cfg.cond_thresh, 10.0));
    double cond_worst = std::max(r.cond_rot, r.cond_trans);
    r.soft_factor = std::max(0.0, std::min(1.0, (denom - std::log10(std::max(cond_worst, 1.0))) / denom));
  }
  return r;
}

// Smallest eigenvalue of the normalised plane-normal scatter matrix (LiDAR).
// Returns -1 for an empty set. If min_dir is non-null it receives the weakest direction.
inline double normalScatterMinEigen(const std::vector<Eigen::Vector3d> &normals, Eigen::Vector3d *min_dir = nullptr)
{
  if (normals.empty()) return -1.0;
  Eigen::Matrix3d N = Eigen::Matrix3d::Zero();
  for (const auto &n : normals) N += n * n.transpose();
  N /= static_cast<double>(normals.size());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(N);
  if (min_dir) *min_dir = es.eigenvectors().col(0);
  return es.eigenvalues()(0);
}

// Stateful detector: analyses one update and applies temporal hysteresis.
class DegeneracyDetector
{
public:
  DegeneracyConfig cfg;
  DegeneracyResult result;

  // HTH: 6x6 pose information block. normals: optional (LiDAR plane normals) for the
  // scatter cross-check, pass nullptr to skip. effective_num: observation count.
  const DegeneracyResult &update(const Eigen::Matrix<double, 6, 6> &HTH, const std::vector<Eigen::Vector3d> *normals, int effective_num)
  {
    result = analyzePoseInformation(HTH, cfg);
    result.effective_num = effective_num;

    if (normals != nullptr)
    {
      Eigen::Vector3d dir;
      result.normal_scatter_min = normalScatterMinEigen(*normals, &dir);
      if (result.normal_scatter_min >= 0.0 && result.normal_scatter_min < cfg.normal_scatter_thresh)
      {
        result.trans_degenerate = true;
        result.trans_degenerate_dir = dir;
      }
    }

    result.degenerate_raw = result.rot_degenerate || result.trans_degenerate;

    if (result.degenerate_raw)
    {
      on_count_++;
      off_count_ = 0;
      if (on_count_ >= cfg.hysteresis_on) latched_ = true;
    }
    else
    {
      off_count_++;
      on_count_ = 0;
      if (off_count_ >= cfg.hysteresis_off) latched_ = false;
    }
    result.degenerate = latched_;
    return result;
  }

private:
  int on_count_ = 0;
  int off_count_ = 0;
  bool latched_ = false;
};

} // namespace degeneracy

#endif // DEGENERACY_H_
