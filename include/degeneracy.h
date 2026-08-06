/*
 * degeneracy.h — 用于 FAST-LIVO2 ESIKF 位姿更新的舒尔补退化检测和
 * 信息域软衰减。
 *
 * 移植并改编自 DCReg（Hu 等，IJRR 2026）：
 *   - 模块 1：基于舒尔补谱分析的退化检测
 *   - 模块 2：物理轴（横滚/俯仰/偏航、x/y/z）表征
 *
 * 此处有意不移植 DCReg 的模块 3（PCG 预条件求解）。
 * FAST-LIVO2 使用误差状态迭代卡尔曼滤波器，其信息形式的更新中已经包含
 * IMU 先验，因此弱方向会受到先验的正则化。这里不采用 DCReg 的“向上钳制
 * 特征值”方法（该方法会使滤波器在不可观测轴上过度自信），而是沿检测到的
 * 弱方向衰减测量信息：这些轴上的卡尔曼增益将趋于零，由 IMU 或其他模态
 * 承担状态估计。
 *
 * DetectPoseDegeneracy() 的输入是仅由测量构成的 6x6 位姿信息矩阵
 * H = J^T R^-1 J，排列顺序为 [rot(0:3), trans(3:6)]，与 FAST-LIVO2 的
 * 误差状态（common_lib.h）以及 DCReg 的海森矩阵布局完全一致。
 *
 * 仅含头文件且仅依赖 Eigen（不依赖 PCL/ROS），可直接包含使用。
 */
#ifndef DEGENERACY_H_
#define DEGENERACY_H_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace degen
{

// 运行时配置。默认关闭此模块，因此配置 yaml 未作修改时，程序行为可保持
// 逐字节一致。
struct DegenParams
{
  bool enable = false;          // 总开关
  bool lidar_enable = true;     // 各模态开关（仅在 enable=true 时使用）
  bool visual_enable = true;
  bool diagnostic_only = false; // 仅检测并记录日志，不修改 H（用于 A/B 对比）
  double cond_threshold = 10.0; // tau：逐轴条件数 lambda_max/lambda_i，超过此值即判定该轴退化
  double gate_floor = 0.0;      // 门控值 g_i 的下限（0 表示可以完全投影掉弱轴）
  bool verbose = false;         // 逐帧记录日志
};

// 一个 6x6 位姿信息矩阵的检测与表征结果。
struct DegenResult
{
  bool ok = false;             // 舒尔分解是否成功
  bool is_degenerate = false;  // 是否有任一轴被标记为退化
  Eigen::Matrix<double, 6, 1> lambda; // 对齐后的逐轴特征值 [横滚,俯仰,偏航,x,y,z]
  Eigen::Matrix<double, 6, 1> gate;   // 逐轴软门控值 g_i，范围为 [gate_floor, 1]
  std::array<bool, 6> mask{};         // 退化掩码 [横滚,俯仰,偏航,x,y,z]
  double cond_rot = 1.0;
  double cond_trans = 1.0;
  Eigen::Matrix3d Vr = Eigen::Matrix3d::Identity(); // 对齐后的旋转特征基
  Eigen::Matrix3d Vt = Eigen::Matrix3d::Identity(); // 对齐后的平移特征基

  DegenResult()
  {
    lambda.setOnes();
    gate.setOnes();
  }
};

// 将原始的（符号及排列不确定的）舒尔特征向量映射到标准坐标轴。
// 移植自 DCReg 的 AlignEigenBasisToAxes（dcreg.hpp）：对每个标准轴，从尚未
// 使用的特征向量中选取内积绝对值最大的一个，然后修正其符号。返回对齐后的
// 特征基和特征值。
inline void AlignEigenBasisToAxes(const Eigen::Matrix3d &raw_basis,
                                  const Eigen::Vector3d &raw_lambda,
                                  Eigen::Matrix3d *aligned_basis,
                                  Eigen::Vector3d *aligned_lambda)
{
  const Eigen::Vector3d refs[3] = {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
                                   Eigen::Vector3d::UnitZ()};
  aligned_basis->setIdentity();
  *aligned_lambda = raw_lambda;
  bool used[3] = {false, false, false};
  for (int axis = 0; axis < 3; ++axis)
  {
    double best_score = -1.0;
    int best_index = -1;
    for (int cand = 0; cand < 3; ++cand)
    {
      if (used[cand]) continue;
      const double score = std::abs(refs[axis].dot(raw_basis.col(cand)));
      if (score > best_score)
      {
        best_score = score;
        best_index = cand;
      }
    }
    if (best_index < 0) best_index = axis; // 防御性处理；正常情况下不会发生
    used[best_index] = true;
    Eigen::Vector3d col = raw_basis.col(best_index);
    if (refs[axis].dot(col) < 0.0) col = -col; // 修正符号
    aligned_basis->col(axis) = col;
    (*aligned_lambda)(axis) = raw_lambda(best_index);
  }
}

// 根据舒尔补填充结果中的 3x3 子块（旋转或平移）：进行特征分解、与坐标轴
// 对齐，并计算逐轴条件数和软门控值。返回该子块使用的 lambda_max。
inline double CharacterizeBlock(const Eigen::Matrix3d &schur, const DegenParams &p,
                                Eigen::Matrix3d *V, Eigen::Vector3d *lambda,
                                Eigen::Vector3d *gate, std::array<bool, 6> *mask,
                                int mask_offset, double *cond, bool *is_degenerate)
{
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(schur);
  Eigen::Matrix3d raw_V = es.eigenvectors();
  Eigen::Vector3d raw_l = es.eigenvalues();
  AlignEigenBasisToAxes(raw_V, raw_l, V, lambda);

  const double lmax = std::max(lambda->maxCoeff(), 1e-12);
  const double lmin = std::max(lambda->minCoeff(), 1e-12);
  *cond = lmax / lmin;
  for (int i = 0; i < 3; ++i)
  {
    const double li = std::max((*lambda)(i), 1e-12);
    const double cond_i = lmax / li;
    // 软门控：条件良好时取 1；超过阈值后，按照 lambda_i/lambda_max 的比例
    // 平滑衰减。
    double g = std::min(1.0, p.cond_threshold * li / lmax);
    g = std::min(1.0, std::max(p.gate_floor, g));
    (*gate)(i) = g;
    const bool degen = cond_i > p.cond_threshold;
    (*mask)[mask_offset + i] = degen;
    if (degen) *is_degenerate = true;
  }
  return lmax;
}

// 模块 1 + 2：检测 6x6 位姿信息矩阵的退化情况，并将弱方向映射到物理轴。
inline DegenResult DetectPoseDegeneracy(const Eigen::Matrix<double, 6, 6> &H_in,
                                        const DegenParams &p)
{
  DegenResult r;
  const Eigen::Matrix<double, 6, 6> H = 0.5 * (H_in + H_in.transpose());

  const Eigen::Matrix3d h_rr = H.block<3, 3>(0, 0);
  const Eigen::Matrix3d h_rt = H.block<3, 3>(0, 3);
  const Eigen::Matrix3d h_tr = H.block<3, 3>(3, 0);
  const Eigen::Matrix3d h_tt = H.block<3, 3>(3, 3);

  const Eigen::FullPivLU<Eigen::Matrix3d> lu_rr(h_rr);
  const Eigen::FullPivLU<Eigen::Matrix3d> lu_tt(h_tt);
  if (!lu_rr.isInvertible() || !lu_tt.isInvertible())
  {
    // 严重秩亏：无法分解特征基，无从判断哪个轴可观，因此将整块测量视为
    // 不可信 —— 门控全部置 0（强制"完全跳过"，即使软衰减的 gate_floor>0
    // 也不放行任何秩亏信息），使 T=0 => H_att=0、HTz=0 => 该模态本帧不更新，
    // 状态完全交给 IMU 先验。对应 DCReg 中 factorization_ok=false 的兜底路径。
    r.ok = false;
    r.is_degenerate = true;
    r.mask = {true, true, true, true, true, true};
    r.gate.setConstant(std::min(1.0, std::max(p.gate_floor, 0.0)));
    r.lambda.setZero();
    r.cond_rot = std::numeric_limits<double>::infinity();
    r.cond_trans = std::numeric_limits<double>::infinity();
    return r;
  }

  // 公式（18）：解耦旋转和平移的可观测性。
  const Eigen::Matrix3d schur_rot = h_rr - h_rt * lu_tt.inverse() * h_tr;
  const Eigen::Matrix3d schur_trans = h_tt - h_tr * lu_rr.inverse() * h_rt;

  Eigen::Vector3d lam_r, lam_t, gate_r, gate_t;
  CharacterizeBlock(schur_rot, p, &r.Vr, &lam_r, &gate_r, &r.mask, 0, &r.cond_rot,
                    &r.is_degenerate);
  CharacterizeBlock(schur_trans, p, &r.Vt, &lam_t, &gate_t, &r.mask, 3, &r.cond_trans,
                    &r.is_degenerate);

  r.lambda.head<3>() = lam_r;
  r.lambda.tail<3>() = lam_t;
  r.gate.head<3>() = gate_r;
  r.gate.tail<3>() = gate_t;
  r.ok = true;
  return r;
}

// 构造对称正定（SPD）衰减算子 T = blkdiag(G_R, G_t)，其中
// G = V * diag(sqrt(g)) * V^T。该公共算子必须一致地应用于测量正规方程的
// 两侧：
//   信息矩阵：H   -> T * H * T
//   信息向量：HTz -> T * HTz      (HTz = H^T R^-1 z)
// 这等价于将测量雅可比软化为 H_eff = H * T。如果只对 H 应用该算子而不处理
// HTz，则近零信息方向上仍会保留完整强度的梯度，从而产生巨大的“增益 × 先验
// 协方差”更新步长并导致滤波器发散。因此，调用方必须使用同一个 T 衰减 HTz。
inline Eigen::Matrix<double, 6, 6> BuildAttenuationOperator(const DegenResult &r)
{
  Eigen::Vector3d sr = r.gate.head<3>().cwiseMax(0.0).cwiseSqrt();
  Eigen::Vector3d st = r.gate.tail<3>().cwiseMax(0.0).cwiseSqrt();
  Eigen::Matrix<double, 6, 6> T = Eigen::Matrix<double, 6, 6>::Zero();
  T.block<3, 3>(0, 0) = r.Vr * sr.asDiagonal() * r.Vr.transpose();
  T.block<3, 3>(3, 3) = r.Vt * st.asDiagonal() * r.Vt.transpose();
  if (!T.allFinite()) T.setIdentity(); // 防御性处理：绝不引入 NaN/Inf
  return T;
}

// 便捷函数：返回 T * H * T（重新对称化）。调用处应优先使用
// BuildAttenuationOperator，以便将同一个 T 同时应用于信息向量 HTz。
inline Eigen::Matrix<double, 6, 6> AttenuateInformation(
    const Eigen::Matrix<double, 6, 6> &H, const DegenResult &r)
{
  const Eigen::Matrix<double, 6, 6> T = BuildAttenuationOperator(r);
  Eigen::Matrix<double, 6, 6> H_att = T * H * T;
  return 0.5 * (H_att + H_att.transpose()); // 重新对称化以抵消舍入误差
}

// 单行诊断信息，格式与 FAST-LIVO2 现有的 "[ LIO ]" 日志保持一致。
inline void Log(const char *tag, const DegenResult &r)
{
  std::cout << "[ DEGEN ][" << tag << "] " << (r.ok ? "ok" : "RANK-DEF")
            << " degenerate=" << (r.is_degenerate ? 1 : 0) << " mask=";
  for (int i = 0; i < 6; ++i) std::cout << (r.mask[i] ? '1' : '0');
  std::cout << " cond_rot=" << r.cond_rot << " cond_trans=" << r.cond_trans
            << " gate=[" << r.gate.transpose() << "] lambda=[" << r.lambda.transpose()
            << "]" << std::endl;
}

} // 命名空间 degen

#endif // DEGENERACY_H_
