import math
import numpy as np
import matplotlib.pyplot as plt


class RobotProp:
    """机器人参数"""
    def __init__(self):
        self.maxV = 0.4
        self.maxW = 1.5
        self.accV = 1.5
        self.accW = 1.5
        self.LeftWheelDiameter = 0.141560990
        self.RightWheelDiameter = 0.142636990
        self.L    = 0.347702681  # 左右轮距
        self.resolV = 0.01  # 速度分辨率
        self.resolW = 0.1  # 角速度分辨率
        self.dt     = 0.1  #
        self.T      = 3.0  # 向前预估时间
        self.kalpha = 0.5
        self.kro = 3.0
        self.kv  = 1.0
        self.weightV = 6.0
        self.weightW = 6.0
        self.weightObs = 1.0


def calc_dynamic_window(x, robot_prop):
    """
    计算动态窗口
    :param: 机器人坐标、机器人参数
    :return: 动态窗口：最小速度，最大速度，最小角速度 最大角速度速度
    """
    # 车子速度的最大最小范围依次为：最小速度 最大速度 最小角速度 最大角速度速度
    vs = [0.03, robot_prop.maxV, -robot_prop.maxW, robot_prop.maxW]

    # 根据当前速度以及加速度限制计算的动态窗口  依次为：最小速度 最大速度 最小角速度 最大角速度速度
    vd = [x[3] - robot_prop.accV * robot_prop.dt,
          x[3] + robot_prop.accV * robot_prop.dt,
          x[4] - robot_prop.accW * robot_prop.dt,
          x[4] + robot_prop.accW * robot_prop.dt]

    # 最终的Dynamic Window
    vr = [max(vs[0], vd[0]), min(vs[1], vd[1]),
          max(vs[2], vd[2]), min(vs[3], vd[3])]

    return vr


def motion(x, u, robot_prop):
    """
    Motion Model 根据当前状态推算下一个控制周期（dt）的状态
    :param x: 机器人当前位置
    :param u: 当前速度
    :param robot_prop: 参数
    :return: 机器人下一刻的位置
    """
    dt = robot_prop.dt
    x[0] += u[0] * math.cos(x[2]) * dt  # x方向位移
    x[1] += u[0] * math.sin(x[2]) * dt  # y
    x[2] += u[1] * dt  # 航向角
    x[3] = u[0]  # 速度v
    x[4] = u[1]  # 角速度w

    return x


def calc_traj(x, v, w, robot_prop):
    """
    轨迹生成
    :param: 机器人坐标，速度，角速度，机器人参数
    :return: 机器人在T时间内的轨迹
    """
    x = np.array(x)
    time = 0
    traj = np.array(x)
    while time <= robot_prop.T:
        time += robot_prop.dt
        x = motion(x, [v, w], robot_prop)
        traj = np.vstack((traj, x))

    return x, traj


def calc_heading(x, goal):
    """
    航向参数得分  当前车的航向和相对于目标点的航向 偏离程度越小 分数越高
    :param x:
    :param goal:
    :return:
    """
    goal_theta = math.atan2(goal[1] - x[1], goal[0] - x[0])
    heading = abs(x[2] - goal_theta)
    dist2goal = abs(math.sqrt((goal[0] - x[0]) ** 2 + (goal[1] - x[1]) ** 2))
    return heading, dist2goal


def calc_score_v_w(alpha, ro, vt, wt, robot_prop):
    """

    :param alpha:
    :param ro:
    :param vt:
    :param wt:
    :param robot_prop:
    :return:
    """
    # 计算线速度得分
    vi = robot_prop.kv * robot_prop.maxV * math.cos(alpha) * math.tanh(ro / robot_prop.kro)
    score_v = 1 - abs(vt - vi) / (2 * robot_prop.maxV)

    # 计算角速度得分
    wi = robot_prop.kalpha * alpha + vi * math.sin(alpha) / ro
    score_w = 1 - abs(wt - wi) / (2 * robot_prop.maxW)
    return score_v, score_w


def calc_score_dis2obs(traj, obs, obs_r):
    """
    障碍物距离评价函数  （机器人在当前轨迹上与最近的障碍物之间的距离，如果没有障碍物则设定一个常数）
    :param traj:
    :param obs:
    :param obs_r:
    :return:
    """
    dis2obs = float("inf")

    # 提取轨迹上的机器人x y坐标
    robotx = traj[:, 0:2]

    for it in range(0, len(robotx[:, 1])):
        for io in range(0, len(obs[:, 0])):
            dx = obs[io, 0] - robotx[it, 0]
            dy = obs[io, 1] - robotx[it, 1]
            disttemp = math.sqrt(dx ** 2 + dy ** 2) - obs_r

            if disttemp < dis2obs:
                dis2obs = disttemp

    # 障碍物距离评价限定一个最大值，如果不设定，一旦一条轨迹没有障碍物，将太占比重
    if dis2obs >= 1.5 * obs_r:
        dis2obs = 1.5 * obs_r

    return dis2obs


def calc_breaking_dist(vt, robot_prop):
    """

    :param vt:
    :param robot_prop:
    :return:
    """
    stopdist = vt ** 2 / (2 * robot_prop.accV)
    return stopdist


def evaluation(x, vr, goal, obs, obs_r, robot_prop):
    """
    评价函数 内部负责产生可用轨迹
    :param x:
    :param vr:
    :param goal:
    :param obs:
    :param obs_r:
    :param robot_prop:
    :return:
    """
    # robot_score = np.array([0, 0, 0, 0, 0])
    robot_score = []
    robot_trajectory = []
    for vt in np.arange(vr[0], vr[1], robot_prop.resolV):
        for wt in np.arange(vr[2], vr[3], robot_prop.resolW):

            # 计算机器人的轨迹
            xt, traj = calc_traj(x, vt, wt, robot_prop)

            # 机器人目标朝向及到目标点距离
            alpha, ro = calc_heading(xt, goal)

            # 机器人线速度及角速度得分
            score_v, score_w = calc_score_v_w(alpha, ro, vt, wt, robot_prop)

            # 机器人障碍物得分
            score_dis2obs = calc_score_dis2obs(traj, obs, obs_r)

            # 机器人刹车距离
            stopdist = calc_breaking_dist(abs(vt), robot_prop)

            if score_dis2obs > stopdist:
                robot_score.append([vt, wt, score_v, score_w, score_dis2obs])
                # robot_score = np.vstack((robot_score, [vt, wt, score_v, score_w, score_dis2obs]))
                robot_trajectory.append(np.transpose(traj))

    robot_score = np.array(robot_score)
    return robot_score, robot_trajectory


def normalization(score):
    """
    归一化处理
    :param score:
    :return:
    """
    if sum(score[:, 2]) != 0:
        score[:, 2] = score[:, 2] / sum(score[:, 2])

    if sum(score[:, 3]) != 0:
        score[:, 3] = score[:, 3] / sum(score[:, 3])

    if sum(score[:, 4]) != 0:
        score[:, 4] = score[:, 4] / sum(score[:, 4])

    return score


def dwa_control(x, goal, obs, obs_r, robot_prop):
    """
    DWA算法实现
    :param x:
    :param goal:
    :param obs:
    :param obs_r:
    :param robot_prop:
    :return:
    """
    score_list = []
    # Dynamic Window: Vr = [vmin, vmax, wmin, wmax] 最小速度 最大速度 最小角速度 最大角速度速度
    # 根据当前状态 和 运动模型 计算当前的参数允许范围
    vr = calc_dynamic_window(x, robot_prop)
    robot_score, robot_trajectory = evaluation(x, vr, goal, obs, obs_r, robot_prop)
    if len(robot_score) == 0:
        print('no path to goal')
        u = np.transpose([0, 0])
    else:
        score = normalization(robot_score)

        for ii in range(0, len(score[:, 0])):
            weights = np.mat([robot_prop.weightV, robot_prop.weightW, robot_prop.weightObs])
            scoretemp = weights * (np.mat(score[ii, 2:5])).T
            score_list.append(scoretemp)

        max_score_id = np.argmax(score_list)
        u = score[max_score_id, 0:2]
        trajectory = robot_trajectory[int(max_score_id)]
        trajectory = np.array(trajectory)
        trajectory = np.transpose(trajectory)

    return u, trajectory


def vw_to_wheel(v, w, l):
    """

    :param v:
    :param w:
    :param l:
    :return:
    """
    vr = v + w * l / 2
    vl = v - w * l / 2
    return vl, vr


def plot_arrow(x, y, yaw, length=0.5, width=0.1):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=1.5 * width, head_width=width)
    plt.plot(x, y)


def draw_dynamic_search(trajectory, best_trajectory, x, goal, ob, obs_r, robot_r):
    """
    画出动态搜索过程图
    :return:
    """
    plt.cla()  # 清除上次绘制图像
    plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], "-g")
    plt.plot(x[0], x[1], "xr")
    plt.plot(0, 0, "og")
    plt.plot(goal[0], goal[1], "ro")
    plt.plot(ob[:, 0], ob[:, 1], "bs")
    plot_arrow(x[0], x[1], x[2])
    plot_circle(ob, obs_r, robot_r)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r')
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.0001)


def plot_circle(obs, obs_r, robot_r):
    for i in range(len(obs[:, 0])):
        theta = np.arange(0, 2*np.pi, 0.01)
        x = obs[i, 0] + obs_r * np.cos(theta)
        y = obs[i, 1] + obs_r * np.sin(theta)
        plt.plot(x, y, color = 'red')
        x_robot = obs[i, 0] + (obs_r + robot_r) * np.cos(theta)
        y_robot = obs[i, 1] + (obs_r + robot_r) * np.sin(theta)
        plt.plot(x_robot, y_robot, color = 'red', linestyle = '--')
