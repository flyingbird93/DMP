import tensorflow as tf
import numpy as np
import random
import cv2
import os
import sys
import Read_Image_batch_tf
import CNN_v3_bn
import ReduceLRplateau
import ReadTFRcord

DIR_images = './../data/ava/images/'
TF_tr = './AVA_train.tfrecord'
TF_ts = './AVA_test.tfrecord'

W_DIR = './weights/RNN_CNN_ts_11_54/RNN_CNN_ts54.ckpt'
Pre_W_DIR = './weights/pre_style_CNN-42/pre_style_CNN.ckpt'
AGG_W_DIR = './weights/RNN_CNN_agg_ts_50/agg_ts50'
CNN_W_DIR = './weights/RNN_CNN_cnn_ts_50/cnn_ts50'
LOG_DIR = './logs/v3_2/1'

batch_size = 64
patch_size = 5
train_epoch = 60
num_class = 2
input_img_size = 256
input_patch_size = 224
num_train_image = 230912
num_test_image = 25088
lr = 0.001
weight_decay = 1e-5
test_num = 1

if len(sys.argv) == 2:
    test_num = sys.argv[1]
    print('{}-{}'.format(sys.argv[0], sys.argv[1]))
"""
Define functions
"""


def get_median(v):
    k_val = 3
    ts = tf.transpose(v, [0, 2, 1])
    tk = tf.nn.top_k(ts, k=k_val).values
    res = []
    for b in range(tk.get_shape()[0]):
        for p in range(tk.get_shape()[1]):
            res.append(tk[b][p][k_val-1])
    res = tf.reshape(res, [int(tk.get_shape()[0]), int(tk.get_shape()[1])])
    return res


def IMG_random_crop(img_org):
    H = img_org.shape[0]
    W = img_org.shape[1]
    # print('H:{} W:{}'.format(H, W))
    short = H if H < W else W
    if short < input_img_size:
        ratio = input_img_size / short
        img_org = cv2.resize(img_org, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

    y = 0 if img_org.shape[0] == input_img_size else random.randrange(0, img_org.shape[0] - input_img_size)
    x = 0 if img_org.shape[1] == input_img_size else random.randrange(0, img_org.shape[1] - input_img_size)

    return img_org[y:y + input_img_size, x:x + input_img_size]


def get_image_patches(img_org):
    bundle = []

    for ite in range(patch_size):
        bundle.append(IMG_random_crop(img_org))

    return bundle


"""
Image Reader
CNN Network
AGG Network
"""

Network = CNN_v3_bn.Network(batch_size=batch_size, patch_size=patch_size, mode=1)


def AggNetwork(input_patches, keep_prob, phase=1):
    print('Aggregation layer')
    with tf.variable_scope('agg_fc_layer') as scope:
        print(np.shape(input_patches))
        # fc1 = CNN_v3_bn.fully_connected_batch(input_patches, 256, name='fc1', phase=phase)
        # fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
        # print(np.shape(fc1))

        layer_min = tf.reduce_min(input_patches, axis=1, keep_dims=False)
        layer_max = tf.reduce_max(input_patches, axis=1, keep_dims=False)
        # layer_med = get_median(fc1)
        # layer_mean = tf.reduce_mean(fc1, axis=1, keep_dims=False)
        print(np.shape(layer_min), np.shape(layer_max))

        layer_concat = tf.concat([layer_min, layer_max], axis=1)
        print(np.shape(layer_concat))

        fc2 = CNN_v3_bn.fully_connected_batch(layer_concat, 256, name='fc2', phase=phase)
        fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)
        print(np.shape(fc2))

        fc3 = CNN_v3_bn.fully_connected(fc2, num_class, name='fc3')
        print(np.shape(fc3))

    return fc3

"""
Define placeholders
"""
image_input_format_1 = tf.placeholder(tf.float32, [batch_size, input_img_size, input_img_size, 3])
image_input_format_2 = tf.placeholder(tf.float32, [batch_size, input_img_size, input_img_size, 3])
image_input_format_3 = tf.placeholder(tf.float32, [batch_size, input_img_size, input_img_size, 3])
image_input_format_4 = tf.placeholder(tf.float32, [batch_size, input_img_size, input_img_size, 3])
image_input_format_5 = tf.placeholder(tf.float32, [batch_size, input_img_size, input_img_size, 3])

train_label = tf.placeholder(tf.int32, [batch_size, ])
train_label_one_hot = tf.reshape(tf.one_hot(train_label, num_class), [-1, num_class])
test_label = tf.placeholder(tf.int32, [batch_size, num_class])

ACC_train = tf.placeholder(tf.float32)
COST_train = tf.placeholder(tf.float32)
ACC_test = tf.placeholder(tf.float32)
AP_test = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
IS_training = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)
"""
Get Feature
"""
chk_Plateau = ReduceLRplateau.ReduceLR(factor=0.1)
fc256_1 = Network.CNN(x=image_input_format_1, keep_prob=keep_prob, phase=IS_training)
fc256_2 = Network.CNN(x=image_input_format_2, keep_prob=keep_prob, reuse=True, phase=IS_training)
fc256_3 = Network.CNN(x=image_input_format_3, keep_prob=keep_prob, reuse=True, phase=IS_training)
fc256_4 = Network.CNN(x=image_input_format_4, keep_prob=keep_prob, reuse=True, phase=IS_training)
fc256_5 = Network.CNN(x=image_input_format_5, keep_prob=keep_prob, reuse=True, phase=IS_training)

expand1 = tf.expand_dims(fc256_1, axis=1)
expand2 = tf.expand_dims(fc256_2, axis=1)
expand3 = tf.expand_dims(fc256_3, axis=1)
expand4 = tf.expand_dims(fc256_4, axis=1)
expand5 = tf.expand_dims(fc256_5, axis=1)

features = tf.concat((expand1, expand2, expand3, expand4, expand5), axis=1)

print(np.shape(features))

logit_batch = AggNetwork(input_patches=features, keep_prob=keep_prob, phase=IS_training)

"""
Set parameters
"""
global_step = tf.Variable(0, trainable=True)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_batch, labels=train_label_one_hot))
print(logit_batch, train_label_one_hot)

all_variables = tf.global_variables()

vars_cnn = [var for var in all_variables if 'classification_net' in var.name]
vars_agg = [var for var in all_variables if 'agg_fc_layer' in var.name]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(
        loss=cost, var_list=all_variables, global_step=global_step)

hypothesis = tf.nn.softmax(logits=logit_batch)
prediction = tf.argmax(hypothesis, 1)
prediction_one_hot = tf.one_hot(prediction, num_class)

ground_truth = tf.argmax(train_label_one_hot, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, ground_truth), tf.float32))

# tr_image, tr_label = img_reader.get_train_batch()
# ts_image, ts_label = img_reader.get_test_batch()
tr_image, tr_label = ReadTFRcord.read_tr_batch(FILE=TF_tr, batch_size=batch_size)
ts_image, ts_label = ReadTFRcord.read_tr_batch(FILE=TF_ts, batch_size=batch_size)


# cnn_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classification_net'))
# agg_saver = tf.train.Saver(var_list=tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='agg_fc_layer'))
saver_all = tf.train.Saver()

for var in all_variables:
    tf.summary.histogram(var.name, var)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    train_cost_summary_e = tf.summary.scalar("Train COST by Epoch", COST_train)
    train_acc_summary_e = tf.summary.scalar("Train Acc by Epoch", ACC_train)
    test_acc_summary = tf.summary.scalar("Test Acc", ACC_test)
    train_cost_summary = tf.summary.scalar("Train Cost", COST_train)
    writer = tf.summary.FileWriter(LOG_DIR)
    writer.add_graph(sess.graph)
    print('Session Start')
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # cnn_saver.restore(sess, Pre_W_DIR)
    # saver_all.restore(sess, W_DIR)
    # cnn_saver.restore(sess, CNN_W_DIR)
    # agg_saver.restore(sess, AGG_W_DIR)

    total_bat = int(num_train_image / batch_size) + 1
    total_bat_ts = int(num_test_image / batch_size) + 1
    print('Total number of batch:TR-{}\tTS-{}'.format(total_bat, total_bat_ts))

    before = 0.90
    threshold = 0.50
    # Training
    for e in range(train_epoch):
        tot_c = 0
        tot_acc = 0
        for b in range(total_bat):
            if b % 200 == 0:
                print('{}/{}'.format(b, total_bat))
            train_imgs, train_labels = sess.run([tr_image, tr_label])
            patch_1 = []
            patch_2 = []
            patch_3 = []
            patch_4 = []
            patch_5 = []
            for i in range(batch_size):
                img_id = train_imgs[i]
                org_img = cv2.imread('{}{}.jpg'.format(DIR_images, img_id))
                patches = get_image_patches(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
                patch_1.append(patches[0])
                patch_2.append(patches[1])
                patch_3.append(patches[2])
                patch_4.append(patches[3])
                patch_5.append(patches[4])
            # tmp_bat = [batch size 5 224 224 224 3]
            # print('tmp_bat: {}'.format(np.shape(patch_1)))

            feed_dict = {image_input_format_1: patch_1, image_input_format_2: patch_2, image_input_format_3: patch_3,
                         image_input_format_4: patch_4, image_input_format_5: patch_5, train_label: train_labels,
                         keep_prob: 0.7, learning_rate: lr, IS_training: 1}

            c, p, gt, acc, _ = sess.run([cost, prediction, ground_truth, accuracy, optimizer], feed_dict=feed_dict)

            tot_c += c/total_bat
            tot_acc += acc/total_bat
            # if b == 10:
            #     print('Cost:{}\nP:{}\nGT:{}\nAcc:{:.5f}'.format(c, p, gt, acc))

            train_cost, step = sess.run([train_cost_summary, global_step],
                                        feed_dict={ACC_train: acc, COST_train: c})

            writer.add_summary(train_cost, step)

        reduce_lr = chk_Plateau.check_Plateau(cur_lr=lr, cur_loss=tot_c)
        if reduce_lr > 0:
            lr = reduce_lr
            print("Epoch-{}\tlr updated ...{}".format(e, lr))

        train_cost_e, train_acc_e, summary = sess.run([train_cost_summary_e, train_acc_summary_e, merged_summary],
                                                      feed_dict={COST_train: tot_c, ACC_train: tot_acc})
        print('Epoch-{}\tCost:{:.5f}\tAccuracy{:.5f}'.format(e, tot_c, tot_acc))
        writer.add_summary(train_cost_e, e)
        writer.add_summary(train_acc_e, e)
        writer.add_summary(summary, e)
        if tot_acc > before:
            save_path = saver_all.save(sess=sess, save_path='./weights/RNN_CNN_tr_{}/RNN_CNN_tr{}.ckpt'.
                                       format(int(tot_acc * 100), int(tot_acc * 100)))
            print('Weights Saved...\n{}'.format(save_path))
            before = tot_acc


    # Test
    print('Test...')
    test_acc = 0
    for b in range(total_bat_ts):
        test_img, ts_labels = sess.run([ts_image, ts_label])
        patch_1 = []
        patch_2 = []
        patch_3 = []
        patch_4 = []
        patch_5 = []
        for i in range(batch_size):
            org_img = cv2.imread('{}{}.jpg'.format(DIR_images, test_img))
            patches = get_image_patches(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))

            patch_1.append(patches[0])
            patch_2.append(patches[1])
            patch_3.append(patches[2])
            patch_4.append(patches[3])
            patch_5.append(patches[4])

        feed_dict = {image_input_format_1: patch_1, image_input_format_2: patch_2,
                     image_input_format_3: patch_3,
                     image_input_format_4: patch_4, image_input_format_5: patch_5,
                     test_label: ts_labels, keep_prob: 1, IS_training: 0}

        acc = sess.run(accuracy, feed_dict=feed_dict)
        test_acc += acc / total_bat_ts

    print('Average Accuracy{:.5f}'.format(test_acc))
    ts_acc = sess.run(test_acc_summary, feed_dict={ACC_test: test_acc})
    writer.add_summary(ts_acc)

    save_path = saver_all.save(sess=sess, save_path='./weights/RNN_CNN2_ts_{}_{}/RNN_CNN2_ts{}.ckpt'.
                               format(test_num, int(test_acc * 100), int(test_acc * 100)))
    print('Weights Saved...\n{}'.format(save_path))

    coord.request_stop()
    coord.join(threads)
