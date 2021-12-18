import pybullet as p
import pybullet_data


def PB_button(button_num, last_num):
    flag = False
    if button_num > last_num:
        flag = True
    else:
        flag = False
    return button_num, flag
