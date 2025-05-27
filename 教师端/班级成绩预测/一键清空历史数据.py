file_path1 = "history pass rate.txt"
file_path2 = "history good rate.txt"
file_path3 = "historyT good num.txt"
file_path4 = "historyT gen num.txt"
file_path5 = "historyT fail num.txt"
file_path6 = "history ave score.txt"
file_path7 = "history max score.txt"
file_path8 = "history min score.txt"

def pass_rate_del():
    with open(file_path1, 'w'):
        pass

def good_rate_del():
    with open(file_path2, 'w'):
        pass

def T_num_del():
    with open(file_path3, 'w'):
        pass
    with open(file_path4, 'w'):
        pass
    with open(file_path5, 'w'):
        pass

def score_del():
    with open(file_path6, 'w'):
        pass
    with open(file_path7, 'w'):
        pass
    with open(file_path8, 'w'):
        pass

user_in = int(input('请选择要清除的数据：'))

if user_in == 1:pass_rate_del()
if user_in == 2:good_rate_del()
if user_in == 3:T_num_del()
if user_in == 4:score_del()
if user_in == 5:
    pass_rate_del()
    good_rate_del()
    T_num_del()
    score_del()

print('已清理！')





