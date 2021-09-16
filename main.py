from NN import Network


def load_data(count=768):
    data_train = []
    data_test = []
    f = open('diabetes3.dt')
    for i in range(count):
        temp_list = []
        line = f.readline()
        line = line.replace('\n', '')
        line = line.split(' ')
        for element in line:
            temp_list.append(float(element))
        if i <= 718:
            data_train.append(temp_list)
        else:
            data_test.append(temp_list)
    return data_train, data_test


def broadcast_data(data, k):
    divided_data = []
    len_data = len(data)
    step = len_data // k
    for i in range(k):
        divided_data.append(data[step * i:step * (i + 1)].copy())
    return prepare_data_for_com(k, divided_data)


def prepare_data_for_com(k, divided_data):
    data_for_com = []
    for i in range(k):
        temp_list = []
        for j in range(len(divided_data)):
            if i != j:
                temp_list += divided_data[j]
        data_for_com.append(temp_list)
    return data_for_com


def loss_for_all(data, networks):
    all_accuracy = []
    for network in networks:
        accuracy_of_net = []
        for i in range(len(data)):
            result, ref = network.feed_forward(test_data=data[i], train=False)
            if ref[0] - result[0][0] < 0.3 and ref[1] - result[0][1] < 0.3:
                accuracy_of_net.append(1)
            else:
                accuracy_of_net.append(0)
        accuracy = sum(accuracy_of_net) * 100 / len(data)
        all_accuracy.append(accuracy)
    return all_accuracy


def loss_for_one(network, data):
    accuracy_of_net = []
    for i in range(len(data)):
        result, ref = network.feed_forward(test_data=data[i], train=False)
        if ref[0] - result[0][0] < 0.3 and ref[1] - result[0][1] < 0.3:
            accuracy_of_net.append(1)
        else:
            accuracy_of_net.append(0)
    accuracy = sum(accuracy_of_net) * 100 / len(data)
    return accuracy


def train(network, len_learn):
    for i in range(len_learn):
        network.feed_forward(train=True)


def start():
    size_of_com = int(input('Введите размерность коммитета '))
    data_train, data_test = load_data()
    data_train = broadcast_data(data_train, size_of_com)
    networks = []
    i = 0
    wrong_nets = 0

    while len(networks) != size_of_com:
        network = Network(8, 2, 5, 2, data_train[i])
        train(network, len(data_train[0])-1)
        accuracy = loss_for_one(network, data_test)
        if accuracy > 50.0:
            networks.append(network)
            i += 1
        else:
            wrong_nets += 1
    accuracy = loss_for_all(data_test, networks)
    common_accuracy = sum(accuracy) / len(accuracy)
    print('Точность каждой сети =', accuracy)
    print('Точность коммитета =', common_accuracy)
    print('Отброшенных сетей', wrong_nets)


if __name__ == '__main__':
    start()
