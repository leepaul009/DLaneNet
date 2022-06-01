import numpy
from collections import OrderedDict 

class hard_sampling():
    def __init__(self):
        self.total_num = 0
        self.first_node = None
        self.last_node = None
        self.minimum_loss = 10000
        self.maximum_size = 1000
        self.ids = list()

    def insert(self, node):
        if node.get_data() not in self.ids:
            # node inserted is not in the buffer
            if self.total_num < 1: # total_num == 0
                self.total_num += 1
                self.first_node = node
                self.last_node = node
                self.minimum_loss = node.get_loss()
                # inserted, then update ids
                self.ids.append(node.get_data())
            else:
                target_loss = node.get_loss()
                # only insert when given loss is bigger than min_loss
                # or when insert full buffer
                if self.minimum_loss < target_loss or self.total_num < self.maximum_size:
                    if self.first_node.get_loss() < target_loss:
                        self.total_num += 1
                        node.set_next(self.first_node)
                        self.first_node.set_previous(node)
                        self.first_node = node
                        # inserted, then update ids
                        self.ids.append(node.get_data())
                    else:
                        current_node = self.first_node             
                        while True:
                            # cur_node >= tar_node, cur_node is last
                            if current_node.get_loss() >= target_loss and current_node.get_next() == None:
                                self.total_num += 1    
                                node.set_previous(current_node)
                                # node is last, its next = None
                                current_node.set_next(node)
                                self.last_node = node 
                                self.minimum_loss = target_loss
                                # inserted, then update ids
                                self.ids.append(node.get_data())
                                break
                            # cur_node >= tar_node >= next_node
                            if current_node.get_loss() >= target_loss and target_loss >= current_node.get_next().get_loss():  
                                self.total_num += 1 
                                node.set_previous(current_node)
                                node.set_next(current_node.get_next())
                                current_node.get_next().set_previous(node)
                                current_node.set_next(node)
                                # inserted, then update ids
                                self.ids.append(node.get_data())
                                break
                            current_node = current_node.get_next()
                            if current_node == None:
                                break
            if self.total_num > self.maximum_size:
                self.total_num -= 1

                # delete last node from self.ids
                del_id = self.last_node.get_data()
                i_to_del = -1
                for i in range(len(self.ids)):
                    if self.ids[i] == del_id:
                        i_to_del = i
                if i_to_del >= 0:
                    del self.ids[i_to_del]
                # end

                self.minimum_loss = self.last_node.get_previous().get_loss()
                self.last_node = self.last_node.get_previous()
                self.last_node.set_next(None)


    def get_list(self):
        data_list = []
        current_node = self.first_node
        while True:
            data_list.append(current_node.get_data())
            current_node = current_node.get_next()
            if current_node == None:
                break
        return data_list

    def get_num(self):
        return self.total_num

'''
class hard_sampler():
    def __init__(self):
        self.total_num = 0
        self.maximum_size = 1000
        self.items = list()
        self.ids = None

    def insert(self, img_id, loss):
        if img_id in self.ids:

        else:
            it = {"img_id":img_id, "loss":loss}
            self.items.append(it)
            self.items = sorted(self.items, key=lambda it: it["loss"], reverse=True)
            if len(self.items) > self.maximum_size:
                del self.items[-1] # less one has min loss
            self.ids = [it['img_id'] for it in self.items]

    def get_ids(self):
        return self.ids
'''

class hard_sampler():
    def __init__(self):
        self.total_num = 0
        self.maximum_size = 1000
        self.items = OrderedDict()

    def insert(self, img_id, loss):
        self.items[img_id] = loss
        # sort as from big to small loss
        self.items = OrderedDict(sorted(self.items.items(), 
                                        key=lambda it: it[1],
                                        reverse=True))
        self.total_num = len(self.items)
        if self.total_num > self.maximum_size:
            # del the smallest loss
            k = list(self.items.items())[-1][0] # get key
            self.items.pop(k)
            self.total_num -= 1

    def get_ids(self):
        return list(self.items.keys())

    def get_num(self):
        return self.total_num

class sampling_node():
    def __init__(self, loss = 10000, data = None, previous_node = None, next_node = None):
        self.loss = loss
        self.data = data
        self.previous_node = previous_node
        self.next_node = next_node

    def set_previous(self, previous_node):
        self.previous_node = previous_node

    def set_next(self, next_node):
        self.next_node = next_node

    def set_loss(self, loss):
        self.loss = loss

    def set_data(self, data):
        self.data = data

    def get_previous(self):
        return self.previous_node

    def get_next(self):
        return self.next_node

    def get_loss(self):
        return self.loss

    def get_data(self):
        return self.data