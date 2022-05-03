import matplotlib.pyplot as plt

x_axis = [ i for i in range(90)] # epoch
y_axis = [] # accuracy
gamma_parameter = [[[],[],[],[],[]] for _ in range(80)]

f = open('CIFAR-10-0.5log.txt', 'r')
lines = f.readlines()
cur_epoch = 0
print_gamma_params = 0
for line in lines:
    item = line.split(" ")
    if 'Epoch:' in item:
        if cur_epoch != int(item[item.index("Epoch:")+1]):
            cur_epoch = int(item[item.index("Epoch:")+1])
            print_gamma_params = 0

    # train accuracy 저장
    if 'Train_acc' in item:
        accuracy = item[item.index("Train_acc")+1]
        accuracy = accuracy[:-2]
        y_axis.append(float(accuracy))
    
    # gamma parameter 저장
    if cur_epoch >= 10:
        if 'The current arch parameters are:' in line:
            # gamma parameter 리스트 저장 flag 1로 set
            print_gamma_params = 1
    
    if print_gamma_params == 1:
        if 'stage:' in item:
            stage_num = int(item[item.index('stage:')+1])
            blk_params = list(item[item.index('block:')+2:])
            blk_params[0] = blk_params[0][1:]
            blk_params = list(map(float, blk_params))
            gamma_parameter[cur_epoch-10][stage_num-1].append(blk_params)

print(len(gamma_parameter))
print(gamma_parameter)         
plt.plot(x_axis, y_axis, 'r-')
# 그래프 제목
plt.title('CIFAR-10 0.5KB')
# x축, y축 이름
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
# 파일로 저장
plt.savefig('cifar-10-0.5.png')