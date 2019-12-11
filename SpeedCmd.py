def calc_speed_cmd(vl, vr):
    # 转16进制
    if vl < 0:
        vl = round((vl*1000).item())
        vl = hex(vl & 0xffffffff)[2:].rjust(8, 'f')
    else:
        vl = str(hex(round((vl * 1000).item()))[2:]).zfill(8)

    if vr < 0:
        vr = round((vr * 1000).item())
        vr = hex(vr & 0xffffffff)[2:].rjust(8, 'f')
    else:
        vr = str(hex(round(vr.item() * 1000))[2:]).zfill(8)

    # 低字节在前
    vl_reverse = vl[6:8] + vl[4:6] + vl[2:4] + vl[0:2]
    vr_reverse = vr[6:8] + vr[4:6] + vr[2:4] + vr[0:2]

    cmd = 'fc0104000000' + vl_reverse + vr_reverse

    # # 计算异或值，负数16进制转10进制有问题：比如ffd就是4093
    # for ii in range(0, int(len(cmd)/2-1)):
    #     if ii == 0:
    #         xor_final = int(cmd[0:2], 16) ^ int(cmd[2:4], 16)
    #     else:
    #         xor_final = xor_final ^ int(cmd[ii*2+2:ii*2+4], 16)

    # 添加fsc
    # speed_cmd = cmd + hex(xor_final)[2:].zfill(2)
    speed_cmd = cmd + '00'
    return speed_cmd
