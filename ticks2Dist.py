import math


def ticks_to_dist(x, y, th, ticks, enc_left, enc_right, wheel_base, l_wheel_d, r_wheel_d):
    """
    转ticks为机器人行走的距离
    :param x:
    :param y:
    :param th:
    :param ticks：
    :param local_left:  m_lCurrLeftTicks: 目前的ticks
    :param local_right: m_lCurrRightTicks
    :param enc_left: 以前的ticks
    :param enc_right:
    :param wheel_base:
    :return:
    """
    # 把接受的ticks剖析：转化成左右轮各自的ticks并重组（原先低位在前）
    raw_left_ticks = ticks[12:20]
    raw_right_ticks = ticks[20:28]
    local_left = int(str(raw_left_ticks[6:8] + raw_left_ticks[4:6] + raw_left_ticks[2:4] + raw_left_ticks[0:2]),
                     16)
    local_right = int(
        str(raw_right_ticks[6:8] + raw_right_ticks[4:6] + raw_right_ticks[2:4] + raw_right_ticks[0:2]), 16)

    # 计算每个ticks是多少距离
    LeftMeterPerTick = l_wheel_d * math.pi / 2000
    RightMeterPerTick = r_wheel_d * math.pi / 2000

    # 按照给的公式计算
    if (enc_left == 0) & (enc_right == 0) & (local_left == 0) & (local_right == 0):
        d_left = d_right = 0
    else:
        d_left  = (local_left - enc_left) * 1.0 * LeftMeterPerTick
        d_right = (local_right - enc_right) * 1.0 * RightMeterPerTick

    # 记录现有的ticks
    enc_left  = local_left
    enc_right = local_right

    tth = (d_right - d_left) / wheel_base  # 变化的角度
    d = (d_left + d_right) / 2.0  # 变化的距离

    dt_ticks = 0.05  # 50毫秒？？？？应该具体计时
    vt = d / dt_ticks
    wt = tth / dt_ticks

    th = th + tth
    x = x + math.cos(th) * d
    y = y + math.sin(th) * d

    return enc_left, enc_right, x, y, th, vt, wt
