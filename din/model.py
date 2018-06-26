import tensorflow as tf

from Dice import dice


class Model(object):

    def __init__(self, user_count, item_count, cate_count, cate_list):

        self.u = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])

        hidden_units = 128

        # user embedding to 128 dim
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        print('------------------------------------------------------------------------------------------------------')
        print('user emb w: ', user_emb_w.get_shape())
        # item embedding to 64 dim
        item_emb_w = tf.get_variable(
            "item_emb_w", [item_count, hidden_units // 2])
        print('item emb w:  ', item_emb_w.get_shape())
        # item b is 63001
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        print('item b :', item_b.get_shape())
        # cate embedding is 64 dim
        cate_emb_w = tf.get_variable(
            "cate_emb_w", [cate_count, hidden_units // 2])
        print('cate_emb_w:', cate_emb_w.get_shape())
        # cate list is 63001
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)
        print('cate list :', cate_list.get_shape())
        # find user u`s embedding
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

        # find item`s category
        ic = tf.gather(cate_list, self.i)
        # cat item and item category embedding together
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
        print('i_emb (refers to going to buy ):',
              i_emb.get_shape())  # [B]

        i_b = tf.gather(item_b, self.i)

        jc = tf.gather(cate_list, self.j)
        # cat bought item and category together
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
        print('j_emb (refers to going to buy):', j_emb.get_shape())
        j_b = tf.gather(item_b, self.j)

        hc = tf.gather(cate_list, self.hist_i)
        # cat history and category
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)
        print('h_emb (refers to already bought, history):', h_emb.get_shape())

        # query keys keylength
        hist = attention(i_emb, h_emb, self.sl)
        # hist is activated h_emb
        # -- attention end ---
        print('hist (? for batch size):', hist.get_shape())
        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, hidden_units])
        hist = tf.layers.dense(hist, hidden_units)

        u_emb = hist
        print('u_emb:', hist.get_shape())
        # print(u_emb.get_shape().as_list())
        # print(i_emb.get_shape().as_list())
        # print(j_emb.get_shape().as_list())
        # -- fcn begin -------

        din_i = tf.concat([u_emb, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(
            din_i, 80, activation=tf.nn.sigmoid, name='f1')
        # if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
        #d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
        #d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
        d_layer_2_i = tf.layers.dense(
            d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        #d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
        d_layer_3_i = tf.layers.dense(
            d_layer_2_i, 1, activation=None, name='f3')

        din_j = tf.concat([u_emb, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(
            inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(
            din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        #d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
        d_layer_2_j = tf.layers.dense(
            d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        #d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
        d_layer_3_j = tf.layers.dense(
            d_layer_2_j, 1, activation=None, name='f3', reuse=True)

        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        # x responsible for AUC
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]

        self.logits = i_b + d_layer_3_i

        u_emb_all = tf.expand_dims(u_emb, 1)
        print('u_emb_all original : ', u_emb_all.get_shape())
        u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])
        print('u_emb_all after tile : ', u_emb_all.get_shape())
        # logits for all item:
        # print(item_emb_w.get_shape())
        all_emb = tf.concat([
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
        print('all_emb shape', all_emb.get_shape())
        all_emb = tf.expand_dims(all_emb, 0)
        print('all_emb after expend: ', all_emb.get_shape())
        all_emb = tf.tile(all_emb, [512, 1, 1])
        print('all emb after tile: ', all_emb.get_shape())
        din_all = tf.concat([u_emb_all, all_emb], axis=-1)
        print('din_all shape: ', din_all.get_shape())
        din_all = tf.layers.batch_normalization(
            inputs=din_all, name='b1', reuse=True)
        d_layer_1_all = tf.layers.dense(
            din_all, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        #d_layer_1_all = dice(d_layer_1_all, name='dice_1_all')
        d_layer_2_all = tf.layers.dense(
            d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        #d_layer_2_all = dice(d_layer_2_all, name='dice_2_all')
        d_layer_3_all = tf.layers.dense(
            d_layer_2_all, 1, activation=None, name='f3', reuse=True)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])
        # self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        # pos and neg
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        # print(self.p_and_n.get_shape().as_list())
        print('----------------------------------------------------------------------------------------------')

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l,
        })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n

    def test(self, sess, uid, hist_i, sl):
        return sess.run(self.logits_all, feed_dict={
            self.u: uid,
            self.hist_i: hist_i,
            self.sl: sl,
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def attention(queries, keys, keys_length):
    """train attention based on history data

    Arguments:
        queries {B,H} -- pos or neg list, things the user going to buy, H for hidden
        keys {B,T,H} -- history data, user already bought, T for time
        keys_length {B} -- history lengths of each sample

    Returns:
        outputs -- where the history data should be activated
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(
        queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
    print('--attention begin--')
    print('query,keys', queries.get_shape(), keys.get_shape())
    print('query-keys', (queries-keys).get_shape())
    print('query*keys', (queries*keys).get_shape())
    print('din_all:', din_all.get_shape())
    print('--attention end--')
    d_layer_1_all = tf.layers.dense(
        din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(
        d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(
        d_layer_2_all, 1, activation=None, name='f3_att')
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all
    print('net output shape:', outputs.get_shape())  # [B, 1, T]
    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    # why paddings?
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs
