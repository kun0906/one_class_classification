"""

    download mawi pcaps from http://mawi.wide.ad.jp/mawi/samplepoint-F/2019/

    A whole month pcap traces
    2019/07: 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
"""
import os
from subprocess import check_call

"""
    https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings
    >>> width = 10
    >>> precision = 4
    >>> value = decimal.Decimal("12.34567")
    >>> f"result: {value:{width}.{precision}}"  # nested fields
    'result:      12.35'

"""


def download_mawi(root_url='http://mawi.nezu.wide.ad.jp/mawi/samplepoint-F/2019'):
    """

    :param root_url:
    :return:
    """
    # url = 'http://mawi.nezu.wide.ad.jp/mawi/samplepoint-F/2019/201907011400.pcap.gz'
    err_cnt = 0
    for idx in range(1, 31 + 1):
        file_name = f'201907{idx:{2}}1400.pcap.gz'  # width = 2, precision = 0
        url = root_url + '/' + file_name
        # url = 'http://mawi.nezu.wide.ad.jp/mawi/samplepoint-F/2019/201907%02d1400.pcap.gz'%idx
        cmd = f"wget {url}"
        print('--> ', cmd)
        if file_name in os.listdir('.'):  # f"result: {value:{width}.{precision}}"  # nested fields
            print(f'idx:{idx}, {file_name} has already downloaded.')
            continue
        try:
            check_call(cmd, shell=True)
        except:
            err_cnt += 1

    print(f'Num. of error download urls: {err_cnt}')


if __name__ == '__main__':
    download_mawi()
