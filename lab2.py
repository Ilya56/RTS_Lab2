from __future__ import print_function

import datetime
from random import uniform

import matplotlib.pyplot as plt
from math import sin, cos, pi, exp

def decor(func):
    def wrapper(*args):
        t_start = datetime.datetime.now()
        res = func(*args)
        t_end = datetime.datetime.now() - t_start
        return res, t_end

    return wrapper


@decor
def generate_signal(ticks_number, lim_freq, harmonics_number):
    result = [0] * ticks_number
    for t in range(ticks_number):
        for i in range(harmonics_number):
            result[t] += uniform(0, 1) * sin(lim_freq * (i / harmonics_number) * t + uniform(0, 1))
    return result


@decor
def mx(data, ticks_number):
    return sum(data) / ticks_number


@decor
def dx(data, mx1, ticks_number):
    if not mx1:
        mx1 = mx(data, ticks_number)
    return sum((data[t] - mx1) ** 2 for t in range(ticks_number)) / (ticks_number - 1)


@decor
def correlate(x_list, y_list, ticks_number):
    r_list = [0 for i in range(len(x_list))]
    Mx, a = mx(x_list, ticks_number)
    My, b = mx(y_list, ticks_number)
    for t in range(len(x_list)):
        r_list[t] = sum((x_list[i] - Mx) * (y_list[i + t] - My) for i in range(len(x_list) - t)) / (len(x_list) - 1)
    return r_list


def do_plot(a_list):
    plt.plot([i for i in range(len(a_list))], a_list)
    plt.axis([0, len(a_list), min(a_list), max(a_list)])
    plt.show()


def iexp(n):
    return complex(cos(n), sin(n))


def DFT(x_n):
    return [sum((x_n[k] * iexp(-2 * pi * i * k / len(x_n)) for k in range(len(x_n)))) for i in range(len(x_n))]


def FFT(n):
    l = len(n)
    if l <= 1:
        return n
    even = FFT(n[0::2])
    odd = FFT(n[1::2])
    t = [iexp(-2 * pi * i / l) * odd[i] for i in range(l // 2)]
    return [even[i] + t[i] for i in range(l // 2)] + [even[i] - t[i] for i in range(l // 2)]


def get_real(arr):
    return [i.real for i in arr]


def get_imaginary(arr):
    return [i.imag for i in arr]


def main():
    harmonics_number = 12
    lim_freq = 2400
    ticks_number = 1024

    x_list, g_res = generate_signal(ticks_number, lim_freq, harmonics_number)
    y_list, g2_res = generate_signal(ticks_number, lim_freq, harmonics_number)

    M_x, m_res = mx(x_list, ticks_number)

    D_x, d_res = dx(x_list, M_x, ticks_number)

    print_data = 'Generate signal time: {4}\nM: {0}, time to calc: {1}\nD: {2}, time to calc: {3}, {4}\n' \
        .format(M_x, m_res, D_x, d_res, g_res, g2_res)

    print(print_data)

    cor1_list, cor1_list_t = correlate(x_list, y_list, ticks_number)
    cor2_list, cor2_list_t = correlate(x_list, x_list, ticks_number)

    print_data += "Correlation\nx_list, y_list: {0}\nx_list, x_list: {1}" \
        .format(cor1_list_t, cor2_list_t)

    dft_res = DFT(x_list)

    fft_res = FFT(x_list)

    plt.subplot(221)
    plt.ylabel('DFT')
    plt.plot(get_real(dft_res))
    plt.subplot(222)
    plt.plot(get_imaginary(dft_res))
    plt.ylabel('FFT')
    plt.subplot(223)
    plt.plot(get_real(fft_res))
    plt.subplot(224)
    plt.plot(get_imaginary(fft_res))

    plt.show()


if __name__ == '__main__':
    main()
