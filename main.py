def load_data(count=768):
    data = []
    f = open('diabetes3.dt')
    for _ in range(count):
        temp_list = []
        line = f.readline()
        line = line.replace('\n', '')
        line = line.split(' ')
        for element in line:
            temp_list.append(float(element))
        data.append(temp_list)
    return data
