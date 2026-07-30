#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mei_to_pinhole.py — 把 MEI(统一全向) 相机的原始图去畸变/校正成针孔透视图。

用于 SubT MRS Hawkins Long Corridor 数据集：
  订阅   /camera_1/image_raw      (sensor_msgs/Image, MEI 原始畸变图)
  发布   /camera_1/image_pinhole  (sensor_msgs/Image, 校正后的针孔透视图)

校正目标针孔内参与 FAST-LIVO2 的 config/camera_SubT_MRS_Hawkins.yaml 完全一致
(fx=fy=300, cx=319.5, cy=239.5, 640x480, 无畸变)，因此 config 里的 Rcl/Pcl 外参
(按 R=I 的校正针孔系标定) 保持有效。

remap 表用 MEI 前向投影(camodocal 约定)一次性预计算，之后每帧只做 cv2.remap，
只需 numpy + cv_bridge + 基础 cv2，不依赖 opencv-contrib(omnidir)。
"""
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# ---- MEI 原始相机内参 (来自数据集标定) ----
XI = 1.507983304667433
K1, K2, P1, P2 = (-0.07347781616721441, 0.24417387274492433,
                  0.0019232361081336904, 0.0014668219651531832)
GAMMA1, GAMMA2 = 740.729341011609, 660.2628898817492
U0, V0 = 319.6316274499249, 235.25773827438545

# ---- 目标针孔内参 (必须与 camera_SubT_MRS_Hawkins.yaml 一致) ----
OUT_W, OUT_H = 640, 480
FX, FY, CX, CY = 300.0, 300.0, 319.5, 239.5


def build_remap():
    """为每个输出针孔像素找到对应的 MEI 源像素坐标, 生成 cv2.remap 用的 map_x/map_y。"""
    # 输出像素网格 -> 相机系归一化射线 (Knew^-1 * [u,v,1])
    us, vs = np.meshgrid(np.arange(OUT_W), np.arange(OUT_H))
    x = (us - CX) / FX
    y = (vs - CY) / FY
    z = np.ones_like(x)

    # 单位球投影 (MEI): 归一化到单位球, 再按镜面参数 xi 平移
    norm = np.sqrt(x * x + y * y + z * z)
    xs, ys, zs = x / norm, y / norm, z / norm
    denom = zs + XI
    xu = xs / denom
    yu = ys / denom

    # 径向 + 切向畸变 (camodocal radtan)
    r2 = xu * xu + yu * yu
    rad = 1.0 + K1 * r2 + K2 * r2 * r2
    xd = xu * rad + 2.0 * P1 * xu * yu + P2 * (r2 + 2.0 * xu * xu)
    yd = yu * rad + P1 * (r2 + 2.0 * yu * yu) + 2.0 * P2 * xu * yu

    # 广义焦距 + 主点 -> 源像素
    map_x = (GAMMA1 * xd + U0).astype(np.float32)
    map_y = (GAMMA2 * yd + V0).astype(np.float32)
    return map_x, map_y


class Rectifier(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.map_x, self.map_y = build_remap()
        self.in_topic = rospy.get_param("~in_topic", "/camera_1/image_raw")
        self.out_topic = rospy.get_param("~out_topic", "/camera_1/image_pinhole")
        self.pub = rospy.Publisher(self.out_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(self.in_topic, Image, self.cb, queue_size=10)
        rospy.loginfo("[mei_to_pinhole] %s -> %s (%dx%d pinhole)",
                      self.in_topic, self.out_topic, OUT_W, OUT_H)

    def cb(self, msg):
        try:
            # 统一转 bgr8 (cv_bridge 会自动处理 bayer/mono/rgb 等原始编码)
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[mei_to_pinhole] convert failed: %s", str(e))
            return
        rect = cv2.remap(img, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR)
        out = self.bridge.cv2_to_imgmsg(rect, encoding="bgr8")
        out.header = msg.header  # 保留时间戳/frame_id, 时间同步很关键
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("mei_to_pinhole")
    Rectifier()
    rospy.spin()
