/*
FAST-LIVO2 的退化检测工具。

本工具自包含且仅依赖 Eigen（不依赖 ROS / PCL），因此可以独立进行单元测试。
它分析 LIO（voxel_map.cpp）和 VIO（vio.cpp）的 IESKF 更新中均已计算的
6x6 位姿信息矩阵（H^T R^-1 H），并报告各方向上的退化情况。

设计说明：
  - 6x6 信息块同时包含旋转（第 0-2 列）和平移（第 3-5 列），二者具有不同的
    物理单位，因此分别分析两个 3x3 子块。这样可使报告的特征方向具有明确的
    物理意义（平移特征向量是世界坐标系中的轴），并允许每个子块使用各自的阈值。
  - 如果子块条件数超过 cond_thresh（相对量、尺度不变，可检测走廊/隧道场景），
    或最小特征值低于绝对下限（用于检测仅靠条件数会漏掉的全局弱可观测性），
    则将相应方向标记为退化。
  - 对于 LiDAR，还可使用一个经过良好归一化的附加交叉检查：匹配平面法向量的
    散布矩阵。其特征值之和为 1（单位法向量），因此最小特征值接近零直接表示
    “没有平面约束该轴”。
  - 使用时间迟滞锁存布尔标志，避免逐帧闪烁。

本模块仅执行检测，绝不修改状态或协方差。
*/

#ifndef DEGENERACY_H_
#define DEGENERACY_H_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace degeneracy
{

struct DegeneracyConfig
{
  bool enable = false;             // 总开关；为 false 时不执行任何分析
  double cond_thresh = 100.0;      // 平移条件数阈值，超过该值时认为该子块退化
  double cond_thresh_rot = 1000.0; // 旋转条件数阈值。旋转信息天然具有各向异性
                                   //（正常运行时 cond_r 通常为 100-200，因为远处点占主导），
                                   // 因此其阈值需要远高于平移阈值，以避免误判旋转退化。
                                   // LiDAR-惯性系统的旋转很少真正退化（受重力和几何约束）。
  double lambda_floor_rot = 1e-3;  // 最强旋转信息特征值的下限；低于该值时子块不含有效信息
  double lambda_floor_trans = 1e-3;// 最强平移信息特征值的下限；低于该值时子块不含有效信息
  double normal_scatter_thresh = 0.02; // 仅用于 LiDAR：归一化平面法向量散布矩阵的最小特征值阈值
  int hysteresis_on = 3;           // 锁存为开启状态所需的连续原始退化帧数
  int hysteresis_off = 3;          // 解除锁存所需的连续正常帧数

  // --- 自适应（自校准）阈值 ---
  // 启用后，平面法向量散布退化测试使用相对于信号自身滑动中位数的阈值
  //（thr = adaptive_alpha * median），而不是固定的 normal_scatter_thresh。
  // 这使检测器基本不受传感器/数据集差异影响：根据经验，手动调参的 LiDAR
  //（Velodyne、Avia）均落在各自中位数的约 0.15-0.18 倍。预热期间（积累足够
  // 样本之前）仍使用 normal_scatter_thresh。仅影响 LiDAR 散布测试（VIO 不传入法向量）。
  bool adaptive = false;
  double adaptive_alpha = 0.15;   // 散布值低于滑动中位数的该比例时判定为退化
  int adaptive_window = 300;      // 基准中位数的滑动窗口长度（帧）
  int adaptive_min_samples = 50;  // 从绝对阈值切换到相对阈值前所需的帧数

  // --- 处理：解重映射（Zhang & Singh 2016）---
  // 启用后，将状态增量中的位姿部分进行投影，以抑制不可观测方向（信息矩阵特征值
  // 远小于最强方向）上的更新，使这些方向回退到 IMU 预测，而不受病态测量更新的
  // 破坏。保持为 false 时仅执行检测。
  bool handling_enable = false;
  double remap_ratio = 100.0;     // 当 eval_max/eval_k 超过该值时抑制对应的平移方向
  bool remap_rot_en = false;      // 同时重映射旋转（默认关闭：旋转很少退化，且针对正常的
                                  // 各向异性进行投影会破坏姿态，进而导致漂移）
  bool soft_remap = true;         // 软衰减（按可观测性缩放）或硬 0/1 投影。
                                  // 软衰减可避免硬投影可能引发的反馈失控。
  bool handling_cov_en = true;    // 保持协方差一致性：同时投影卡尔曼增益 G，使 P=(I-G)P
                                  // 在退化方向上保留先验不确定性（避免滤波器在这些方向上
                                  // 过度自信，从而在离开走廊时更快地重新定位）

  // VIO 专用：通过跟踪点数量（尺度无关、含义直接且已计算）检测整体“VIO 弱/盲”，
  // 并相对于其滑动中位数进行自适应。该方法取代 VIO 中不可移植的绝对特征值下限
  //（光度 H^T H 具有任意且依赖场景的尺度）。VIO 的方向性退化仍由无量纲条件数判断。
  double vio_adaptive_alpha = 0.3; // 当 total_points < alpha * 滑动中位数时标记 VIO 为弱
  int vio_min_points = 15;         // 预热阶段/绝对回退情况下的点数下限
  bool vio_handling_en = false;    // 对 VIO 更新本身应用软衰减。默认关闭：经验表明这会适得其反——
                                   // 跨模态补偿需要 VIO 在 LiDAR 退化方向上的完整更新；抑制 VIO
                                   // 会移除这种补偿并造成漂移。VIO 也会通过先验自正则化。
                                   // 此选项仅为研究用途而保留。
};

struct DegeneracyResult
{
  // 分块特征分析（特征值升序排列；索引 0 对应最弱方向）。
  Eigen::Vector3d eval_rot = Eigen::Vector3d::Zero();
  Eigen::Vector3d eval_trans = Eigen::Vector3d::Zero();
  double cond_rot = 1.0;
  double cond_trans = 1.0;
  bool rot_degenerate = false;
  bool trans_degenerate = false;
  Eigen::Vector3d rot_degenerate_dir = Eigen::Vector3d::Zero();   // 最弱旋转轴（状态切空间）
  Eigen::Vector3d trans_degenerate_dir = Eigen::Vector3d::Zero(); // 最弱平移轴（世界坐标系）

  // 辅助指标。
  double normal_scatter_min = -1.0; // LiDAR 平面法向量散布的最小特征值；未计算时为 -1
  int effective_num = 0;            // 本次更新使用的观测数量
  double scatter_baseline = -1.0;    // 散布值的滑动中位数（自适应模式；未启用/未就绪时为 -1）
  double scatter_thresh_used = -1.0; // 本帧采用的有效散布阈值（无散布值时为 -1）

  // 综合判定结果。
  bool degenerate_raw = false; // 瞬时结果（当前帧）
  bool degenerate = false;     // 经迟滞处理后的锁存结果
  double soft_factor = 1.0;    // 连续置信度：0 = 完全退化，1 = 条件良好
};

// 分析 6x6 位姿信息块。第 0-2 行/列为旋转，第 3-5 行/列为平移。
inline DegeneracyResult analyzePoseInformation(const Eigen::Matrix<double, 6, 6> &HTH, const DegeneracyConfig &cfg)
{
  DegeneracyResult r;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot(HTH.block<3, 3>(0, 0));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_trans(HTH.block<3, 3>(3, 3));
  r.eval_rot = es_rot.eigenvalues();     // 升序
  r.eval_trans = es_trans.eigenvalues(); // 升序
  r.rot_degenerate_dir = es_rot.eigenvectors().col(0);
  r.trans_degenerate_dir = es_trans.eigenvectors().col(0);

  const double eps = 1e-12;
  double lr_min = std::max(r.eval_rot(0), 0.0);
  double lr_max = std::max(r.eval_rot(2), 0.0);
  double lt_min = std::max(r.eval_trans(0), 0.0);
  double lt_max = std::max(r.eval_trans(2), 0.0);
  r.cond_rot = lr_max / std::max(lr_min, eps);
  r.cond_trans = lt_max / std::max(lt_min, eps);

  // 如果子块基本不包含任何信息（其最强特征值低于下限），或者具有强各向异性
  //（条件数高于阈值，即某些方向远弱于观测最充分的方向），则认为该子块退化。
  // 对“无信息”情况检测 lambda_max（而非 lambda_min）更为稳健：最小特征值在数值上
  // 波动很大（只要任一方向不受约束就可能约等于 0），而最大特征值较为稳定。
  bool rot_has_info = (lr_max >= cfg.lambda_floor_rot);
  bool trans_has_info = (lt_max >= cfg.lambda_floor_trans);
  r.rot_degenerate = !rot_has_info || (r.cond_rot > cfg.cond_thresh_rot);
  r.trans_degenerate = !trans_has_info || (r.cond_trans > cfg.cond_thresh);

  // [0,1] 范围内的连续置信度：根据每个子块各自的条件数阈值进行对数缩放
  //（旋转和平移天然具有截然不同的各向异性），然后取两者中较差的结果。
  // 任一子块不含有效信息时取 0。
  if (!rot_has_info || !trans_has_info)
  {
    r.soft_factor = 0.0;
  }
  else
  {
    double dt = std::log10(std::max(cfg.cond_thresh, 10.0));
    double dr = std::log10(std::max(cfg.cond_thresh_rot, 10.0));
    double soft_t = (dt - std::log10(std::max(r.cond_trans, 1.0))) / dt;
    double soft_r = (dr - std::log10(std::max(r.cond_rot, 1.0))) / dr;
    r.soft_factor = std::max(0.0, std::min(1.0, std::min(soft_t, soft_r)));
  }
  return r;
}

// 3x3 信息块的投影/加权矩阵，用于衰减状态增量在弱观测特征方向上的分量。
// 支持两种模式：
//   soft=false（硬重映射）：移除（置零）所有满足 eval_max/eval_k > remap_ratio 的方向。
//     方法简单，但硬性的 0/1 跳变在某些场景中可能引发反馈失控
//     （投影 -> 漂移 -> 匹配变差 -> 进一步投影）。
//   soft=true（默认）：按 w = eval_k/(eval_k + eval_max/remap_ratio) 缩放每个方向，
//     从 1（观测充分）平滑过渡到 0（不可观测）。在弱方向上保留部分 LiDAR 信息，
//     避免硬投影的不稳定性，类似于 Tikhonov 正则化。
// 将结果应用于误差状态增量，可衰减弱方向上的校正量。
// removed（可选）：受到有效抑制的方向数量（软模式中 w<0.5；硬模式中为被投影的方向）。
inline Eigen::Matrix3d degenerateProjection(const Eigen::Matrix3d &info_block, double remap_ratio, bool soft, int *removed = nullptr)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(info_block);
  Eigen::Vector3d ev = es.eigenvalues(); // 升序
  double lmax = std::max(ev(2), 1e-12);
  double reg = lmax / std::max(remap_ratio, 1.0);
  Eigen::Matrix3d P;
  if (soft) { P.setZero(); }
  else { P.setIdentity(); }
  int rm = 0;
  for (int k = 0; k < 3; ++k)
  {
    Eigen::Vector3d v = es.eigenvectors().col(k);
    double lk = std::max(ev(k), 0.0);
    if (soft)
    {
      double w = lk / (lk + reg); // 范围为 [0,1]
      P += w * v * v.transpose();
      if (w < 0.5) ++rm;
    }
    else if (lmax / std::max(lk, 1e-12) > remap_ratio)
    {
      P -= v * v.transpose();
      ++rm;
    }
  }
  if (removed) *removed = rm;
  return P;
}

// 归一化平面法向量散布矩阵（LiDAR）的最小特征值。
// 集合为空时返回 -1。如果 min_dir 非空，则通过它返回最弱方向。
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

// 用于跟踪标量信号中位数的定长滑动窗口（用于自适应阈值）。
class RollingBaseline
{
public:
  void setWindow(size_t w) { window_ = std::max<size_t>(1, w); }
  void push(double v)
  {
    buf_.push_back(v);
    while (buf_.size() > window_) buf_.pop_front();
  }
  size_t size() const { return buf_.size(); }
  double median() const
  {
    if (buf_.empty()) return -1.0;
    std::vector<double> tmp(buf_.begin(), buf_.end());
    size_t mid = tmp.size() / 2;
    std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
    return tmp[mid];
  }

private:
  std::deque<double> buf_;
  size_t window_ = 300;
};

// 有状态检测器：分析一次更新并应用时间迟滞。
class DegeneracyDetector
{
public:
  DegeneracyConfig cfg;
  DegeneracyResult result;

  // HTH：6x6 位姿信息块。normals：用于散布交叉检查的可选参数（LiDAR 平面法向量），
  // 传入 nullptr 可跳过检查。effective_num：观测数量。
  const DegeneracyResult &update(const Eigen::Matrix<double, 6, 6> &HTH, const std::vector<Eigen::Vector3d> *normals, int effective_num)
  {
    result = analyzePoseInformation(HTH, cfg);
    result.effective_num = effective_num;

    if (normals != nullptr)
    {
      Eigen::Vector3d dir;
      result.normal_scatter_min = normalScatterMinEigen(*normals, &dir);
      if (result.normal_scatter_min >= 0.0)
      {
        // 有效散布阈值：自适应模式完成预热后使用相对于滑动中位数的阈值，
        // 否则使用固定的绝对阈值 normal_scatter_thresh。
        double thr = cfg.normal_scatter_thresh;
        if (cfg.adaptive)
        {
          scatter_baseline_.setWindow(static_cast<size_t>(std::max(1, cfg.adaptive_window)));
          scatter_baseline_.push(result.normal_scatter_min);
          if (static_cast<int>(scatter_baseline_.size()) >= cfg.adaptive_min_samples)
          {
            result.scatter_baseline = scatter_baseline_.median();
            thr = cfg.adaptive_alpha * result.scatter_baseline;
          }
        }
        result.scatter_thresh_used = thr;
        if (result.normal_scatter_min < thr)
        {
          result.trans_degenerate = true;
          result.trans_degenerate_dir = dir;
        }
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

  // VIO 更新：无量纲条件数（方向性退化、尺度不变）+ 自适应跟踪点数测试
  //（整体 VIO 弱/盲）。这里刻意不依赖绝对特征值下限（将 VIO 的 lambda_floor
  // 设得很小，使 analyzePoseInformation 的有效信息检查不起作用）；光度信息矩阵的
  // 尺度是任意且依赖场景的，因此其绝对下限不具备可移植性。total_points 尺度无关、
  // 含义直接，并可通过其中位数进行自校准。
  const DegeneracyResult &updateVisual(const Eigen::Matrix<double, 6, 6> &HTH, int total_points)
  {
    result = analyzePoseInformation(HTH, cfg); // 基于条件数的方向性退化 + soft_factor
    result.effective_num = total_points;

    double thr = static_cast<double>(cfg.vio_min_points); // 预热阶段/非自适应回退值
    if (cfg.adaptive)
    {
      pts_baseline_.setWindow(static_cast<size_t>(std::max(1, cfg.adaptive_window)));
      pts_baseline_.push(static_cast<double>(total_points));
      if (static_cast<int>(pts_baseline_.size()) >= cfg.adaptive_min_samples)
      {
        result.scatter_baseline = pts_baseline_.median();        // 复用字段：点数的滑动中位数
        thr = cfg.vio_adaptive_alpha * result.scatter_baseline;
      }
    }
    result.scatter_thresh_used = thr;                            // 复用字段：有效点数阈值
    if (static_cast<double>(total_points) < thr)                // 整体 VIO 弱/盲 -> 所有方向
    {
      result.trans_degenerate = true;
      result.rot_degenerate = true;
      result.soft_factor = 0.0;
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
  RollingBaseline scatter_baseline_;
  RollingBaseline pts_baseline_;
};

} // 命名空间 degeneracy

#endif // DEGENERACY_H_ 结束
