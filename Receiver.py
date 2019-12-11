import threading
import numpy as np
import ticks2Dist
import codecs
import DWA_Speed as dwa
import SpeedCmd
import socket    # 引入套接字
import math
import time


def udp_receiver(s, lock):
    """
    接收ticks模块
    :param s:
    :param lock:
    :return:
    """
    t = threading.currentThread()  # 建立一个线程
    global all_ticks  # 全局化ticks
    all_ticks = []
    while True:
        # 接受ticks
        ticks, addr = s.recvfrom(1024)
        # 记录ticks并加锁
        run_time = time.time()
        lock.acquire()
        all_ticks.append([ticks, run_time])
        lock.release()


def localization(x0, enc_left, enc_right, history_x, robot_prop, lock):
    """
    定位模块，根据ticks定位
    :param x0:
    :param enc_left:
    :param enc_right:
    :param history_x:
    :param robot_prop:
    :param lock:
    :return:
    """
    # 读取记录的ticks
    lock.acquire()
    global all_ticks
    current_ticks = all_ticks
    all_ticks = []
    lock.release()
    # 定位
    for i in range(len(current_ticks)):
        ticks = codecs.encode(current_ticks[i][0], 'hex_codec')
        ticks = str(ticks, encoding='utf8')

        # 根据ticks变化和上次机器人的位置定位
        enc_left, enc_right, robotx, roboty, robotth, vt, wt = ticks2Dist.ticks_to_dist(x0[0], x0[1], x0[2], ticks,
                                                                                        enc_left, enc_right,
                                                                                        robot_prop.L,
                                                                                        robot_prop.LeftWheelDiameter,
                                                                                        robot_prop.RightWheelDiameter)
        x0 = np.array([robotx, roboty, robotth, vt, wt])

        # 历史轨迹的保存
        history_x = np.vstack((history_x, x0))

    return enc_left, enc_right, history_x, x0


def udp_planner(s, x0, goal, obs, sum_r, robot_prop):
    """
    规划速度
    :param s:
    :param x0:
    :param goal:
    :param obs:
    :param sum_r:
    :param robot_prop:
    :return:
    """
    t = threading.currentThread()
    # DWA参数输入 返回控制量 u = [v(m/s),w(rad/s)] 和 轨迹
    u, trajectory = dwa.dwa_control(x0, goal, obs, sum_r, robot_prop)
    # 线速度转轮速度
    vl, vr = dwa.vw_to_wheel(u[0], u[1], robot_prop.L)
    # 速度转16进制
    speed_cmd = SpeedCmd.calc_speed_cmd(vl, vr)
    speed_cmd = str(speed_cmd)
    # 发送速度指令
    print('速度', vl, vr)
    print('sudu', speed_cmd)
    s.sendto(codecs.decode(speed_cmd, 'hex_codec'), ('192.168.31.201', 22001))
    return u, trajectory


def main():
    """
    主程序
    :return:
    """
    global all_ticks
    # 搭建一个server并规定ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('192.168.31.136', 21001))

    # 创建一个互斥锁
    lock = threading.Lock()

    # 初始化机器人位置
    x0 = np.array([5.0, 0.0, math.pi / 2, 0.0, 0.0])
    history_x = np.array(x0)
    enc_left = enc_right = 0
    robot_prop = dwa.RobotProp()

    # 目标点及障碍物信息
    goal = np.array([5.0, 10.0])
    obs = np.mat([[6.0, 4.0],
                  [3.5, 4.5],
                  [8.0, 4.0]])
    obs_r = 0.5
    robot_r = robot_prop.L / 2
    sum_r = obs_r + robot_r

    # 运行服务器线程
    receiver = threading.Thread(target=udp_receiver, args=(s,lock), name='receiver')
    receiver.start()
    id = 0
    while True:
        u, trajectory = udp_planner(s, x0, goal, obs, sum_r, robot_prop)
        enc_left, enc_right, history_x, x0 = localization(x0, enc_left, enc_right, history_x, robot_prop, lock)
        if id > 5:
            dwa.draw_dynamic_search(history_x, trajectory, x0, goal, obs, obs_r, robot_r)
            # 是否到达目的地
            if math.sqrt((x0[0] - goal[0]) ** 2 + (x0[1] - goal[1]) ** 2) <= 0.5:
                speed_cmd = SpeedCmd.calc_speed_cmd(np.float64(0.0), np.float64(0.0))
                speed_cmd = str(speed_cmd)
                s.sendto(codecs.decode(speed_cmd, 'hex_codec'), ('192.168.31.201', 22001))
                print('Arrive Goal!!!')
                break
        id += 1


if __name__ == '__main__':
    main()
