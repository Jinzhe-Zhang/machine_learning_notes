import numpy
import sys

### 定义函数 conv_ 即卷积操作中，具体的矩阵运算方式：
### 总的来说，conv_ 函数的实现内容就是用filter在img上进行遍历，并在每一步进行卷积（矩阵内积），并获得相应结果值的过程
def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = numpy.zeros(img.shape) # 按 img 的尺寸构建结果
    #Looping through the image to apply the convolution operation.


    ### 注意后面定义的 conv 函数，几个判断要求保证了输入的 filter 是正矩阵
    ### 实际上，就是以filter（矩阵）的中心为基点(r,c)，在原图 img 上进行遍历的过程，该基点能够达到的位置

    for r in numpy.uint16(numpy.arange(filter_size/2.0,
                          img.shape[0]-filter_size/2.0+1)):
        for c in numpy.uint16(numpy.arange(filter_size/2.0,
                                           img.shape[1]-filter_size/2.0+1)):

            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on
            the image and filer sizes is the most tricky part of convolution.
            """

            ### 即取出与filter尺寸相同的 img 内容。'

            curr_region = img[r-numpy.uint16(numpy.floor(filter_size/2.0)):r+numpy.uint16(numpy.ceil(filter_size/2.0)),
                              c-numpy.uint16(numpy.floor(filter_size/2.0)):c+numpy.uint16(numpy.ceil(filter_size/2.0))]

            #Element-wise multipliplication between the current region and the filter.
            ### 将对应的内容（filter 与 img 对应filter尺寸的部分）进行矩阵相乘——即卷积操作
            curr_result = curr_region * conv_filter
            conv_sum = numpy.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            ### 进一步将所有求和的值，存储在 result 中，result是一个矩阵，尺寸即为filter 中心‘走’过的座标位置，

    #Clipping the outliers of the result matrix.
    ### 这一步将只保留 result 中有用的部分（即存储 numpy.sum 值的部分），

    final_result = result[numpy.uint16(filter_size/2.0):result.shape[0]-numpy.uint16(filter_size/2.0),
                          numpy.uint16(filter_size/2.0):result.shape[1]-numpy.uint16(filter_size/2.0)]
    return final_result


def conv(img, conv_filter):  ### 这个函数是调整并设置卷积计算的各个要素的维度和尺寸，并通过引入 conv_ 进行计算
    ###-------------------------------------------------------------------------------
    ### 该函数相对 conv_ 而言，是对多重或多维（多通道）的 img 或 filter 制定相应的操作规则，
    ### 而 conv_ 是按照这个规则，进行一对一的卷积操作运算
    ###-------------------------------------------------------------------------------

    if len(img.shape) > 2 or len(conv_filter.shape) > 3:  # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()

    if conv_filter.shape[1] != conv_filter.shape[2]:  # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()

    if conv_filter.shape[1] % 2 == 0:  # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    ### 但要注意，通过filter所得结果的结构形式构造，与传统理解上做了个小变化，将filter的通道数目列在了最后（filter.shape[0]），
    ### 这将在 numpy.zeros 的构造中建立以后两个数为矩阵维度、第一个数为组数的三维矩阵，
    ### 后面的操作将第三个维度作为标识，用前两个维度构成的结构存储了 conv_map，即卷积后的结果，

    feature_maps = numpy.zeros((img.shape[0] - conv_filter.shape[1] + 1,
                                img.shape[1] - conv_filter.shape[1] + 1,
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):  ### 对于每个通道上的 filter
        print("Filter ", filter_num + 1)

        ### 定下的第一个值作为通道后，使用第二维以后的内容作为filter矩阵进行操作
        curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.
        """
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:  ### 如果所选择的当前通道filter上的维度数目大于2
            ### 则采用当前通道上，各组矩阵的第一列转置为行后，构成的矩阵（维度是 [:,:] ），
            ### 并使用该矩阵与相应的图形（注意按判断1， 维度相同）进行具体的卷积操作 “conv_”
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])  # Array holding the sum of all feature maps.

            ### 对于各个子通道的filter
            for ch_num in range(1, curr_filter.shape[
                -1]):  # Convolving each channel with the image and summing the results.

                conv_map = conv_map + conv_(img[:, :, ch_num],
                                            curr_filter[:, :, ch_num])
        else:  # There is just a single channel in the filter.
            ### 否则，如果只是单一个通道，直接 conv_ 操作就可以了
            conv_map = conv_(img, curr_filter)

        ### 这里用到了上面的 feature_maps 的情况：每个通道单独给一个 conv_map 卷积结果数值

        feature_maps[:, :, filter_num] = conv_map  # Holding feature map with the current filter.
    ###返回 feature_maps
    return feature_maps  # Returning all feature maps.

def pooling(feature_map, size=2, stride=2):
    #Preparing the output of the pooling operation.
    ### 最大池化，直接制定了默认池化尺寸是 2x2 ，

    pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0]-size+1)/stride),
                            numpy.uint16((feature_map.shape[1]-size+1)/stride),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in numpy.arange(0,feature_map.shape[0]-size-1, stride):
            c2 = 0
            for c in numpy.arange(0, feature_map.shape[1]-size-1, stride):
                ### 在通道 map_num 上，池化矩阵第(r2,c2)的位置上，做最大池化的操作，即选择该池化尺寸对应的f_map内容中的最大值，

                pool_out[r2, c2, map_num] = numpy.max([feature_map[r:r+size,  c:c+size]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out

def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    ### ReLU是激励函数中取 max(0,value) 的实现方法
    relu_out = numpy.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0,feature_map.shape[0]):
            for c in numpy.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
    return relu_out

if __name__ == '__main__':
    import skimage.data
    import matplotlib.pyplot as plt
    #import numpy
    #导入到notebook中，原.py文件导入不再需要：import numpycnn， 以及 import numpy
    img = skimage.data.chelsea()
    # Converting the image into gray.
    ### 转化成灰度图
    img = skimage.color.rgb2gray(img)
    # First conv layer
    #l1_filter = numpy.random.rand(2,7,7)*20 # Preparing the filters randomly.
    ### 定义第一层 filter 的结构形式，为2层（通道）的 3x3矩阵
    l1_filter = numpy.zeros((2,3,3))
    ### 量化第一个通道内容
    l1_filter[0, :, :] = numpy.array([[[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]]])
    ### 量化第二个通道内容
    l1_filter[1, :, :] = numpy.array([[[1,   1,  1],
                                       [0,   0,  0],
                                       [-1, -1, -1]]])

    print("\n**Working with conv layer 1**")

    ### 使用 conv 进行第一层的卷积计算
    l1_feature_map = conv(img, l1_filter)
    print("\n**ReLU**")
    #l1_feature_map_relu = numpycnn.relu(l1_feature_map)
    ### 对第一层卷积层计算结果应用 ReLU 函数
    l1_feature_map_relu = relu(l1_feature_map)
    print("\n**Pooling**")
    #l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
    ### 对第一层卷积层计算结果进行 Pooling 处理
    l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)
    print("**End of conv layer 1**\n")

    ### 至此，由于filter矩阵维度是(2,3,3)，即2通道3x3矩阵，因此经过卷积操作(conv)的矩阵维度是 2通道，每个通道(n-(3-1))x(m-(3-1))，即，298x449
    ### 激励函数 ReLU 只是对卷积层结果的每个值进行计算变换，因此维度不变，依然是(2x298x449)
    ### 池化层采用 2x2结构，对两个通道的每个结果进行处理，因此仍为2组，每组为 (n'/2)x(m'/2)=(148x224)
    # Second conv layer
    ### 构造第二层的filter，结构上使用上一层(l1)的最终维度
    ### 按照 numpy.random.rand 的方法实现，这个是一个5x2矩阵，分了3大组，每组5个矩阵
    l2_filter = numpy.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
    print("\n**Working with conv layer 2**")
    #l2_feature_map = numpycnn.conv(l1_feature_map_relu_pool, l2_filter)
    ###
    l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
    print("\n**ReLU**")
    #l2_feature_map_relu = numpycnn.relu(l2_feature_map)
    l2_feature_map_relu = relu(l2_feature_map)
    print("\n**Pooling**")
    #l2_feature_map_relu_pool = numpycnn.pooling(l2_feature_map_relu, 2, 2)
    l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)
    print("**End of conv layer 2**\n")
    # Third conv layer
    l3_filter = numpy.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
    print("\n**Working with conv layer 3**")
    #l3_feature_map = numpycnn.conv(l2_feature_map_relu_pool, l3_filter)
    l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
    print("\n**ReLU**")
    #l3_feature_map_relu = numpycnn.relu(l3_feature_map)
    l3_feature_map_relu = relu(l3_feature_map)
    print("\n**Pooling**")
    #l3_feature_map_relu_pool = numpycnn.pooling(l3_feature_map_relu, 2, 2)
    l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
    print("**End of conv layer 3**\n")
    # Graphing results
    ### 使用 matplotlib 进行绘图，本cell先绘制原图 in_img.png，具体细节见 matplotlib mannal，这里不细述
    fig0, ax0 = plt.subplots(nrows=1, ncols=1)
    ax0.imshow(img).set_cmap("gray")
    ax0.set_title("Input Image")
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    plt.savefig("in_img.png", bbox_inches="tight")
    plt.close(fig0)
    # Layer 1
    ### 绘制第一层卷积、ReLU处理、池化后的图，因该层 filter 是两通道，因此每个（conv/ReLU/pooling）结果是两个图
    fig1, ax1 = plt.subplots(nrows=3, ncols=2)
    ax1[0, 0].imshow(l1_feature_map[:, :, 0]).set_cmap("gray")
    ax1[0, 0].get_xaxis().set_ticks([])
    ax1[0, 0].get_yaxis().set_ticks([])
    ax1[0, 0].set_title("L1-Map1")

    ax1[0, 1].imshow(l1_feature_map[:, :, 1]).set_cmap("gray")
    ax1[0, 1].get_xaxis().set_ticks([])
    ax1[0, 1].get_yaxis().set_ticks([])
    ax1[0, 1].set_title("L1-Map2")

    ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax1[1, 0].get_xaxis().set_ticks([])
    ax1[1, 0].get_yaxis().set_ticks([])
    ax1[1, 0].set_title("L1-Map1ReLU")

    ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1]).set_cmap("gray")
    ax1[1, 1].get_xaxis().set_ticks([])
    ax1[1, 1].get_yaxis().set_ticks([])
    ax1[1, 1].set_title("L1-Map2ReLU")

    ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax1[2, 0].get_xaxis().set_ticks([])
    ax1[2, 0].get_yaxis().set_ticks([])
    ax1[2, 0].set_title("L1-Map1ReLUPool")

    ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
    ax1[2, 0].get_xaxis().set_ticks([])
    ax1[2, 0].get_yaxis().set_ticks([])
    ax1[2, 1].set_title("L1-Map2ReLUPool")

    plt.savefig("L1.png", bbox_inches="tight")
    plt.close(fig1)
    # Layer 2
    ### 绘制第二层卷积、ReLU处理、池化后的图，因该层 filter 是三通道，因此每个（conv/ReLU/pooling）结果是三个图
    fig2, ax2 = plt.subplots(nrows=3, ncols=3)
    ax2[0, 0].imshow(l2_feature_map[:, :, 0]).set_cmap("gray")
    ax2[0, 0].get_xaxis().set_ticks([])
    ax2[0, 0].get_yaxis().set_ticks([])
    ax2[0, 0].set_title("L2-Map1")

    ax2[0, 1].imshow(l2_feature_map[:, :, 1]).set_cmap("gray")
    ax2[0, 1].get_xaxis().set_ticks([])
    ax2[0, 1].get_yaxis().set_ticks([])
    ax2[0, 1].set_title("L2-Map2")

    ax2[0, 2].imshow(l2_feature_map[:, :, 2]).set_cmap("gray")
    ax2[0, 2].get_xaxis().set_ticks([])
    ax2[0, 2].get_yaxis().set_ticks([])
    ax2[0, 2].set_title("L2-Map3")

    ax2[1, 0].imshow(l2_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax2[1, 0].get_xaxis().set_ticks([])
    ax2[1, 0].get_yaxis().set_ticks([])
    ax2[1, 0].set_title("L2-Map1ReLU")

    ax2[1, 1].imshow(l2_feature_map_relu[:, :, 1]).set_cmap("gray")
    ax2[1, 1].get_xaxis().set_ticks([])
    ax2[1, 1].get_yaxis().set_ticks([])
    ax2[1, 1].set_title("L2-Map2ReLU")

    ax2[1, 2].imshow(l2_feature_map_relu[:, :, 2]).set_cmap("gray")
    ax2[1, 2].get_xaxis().set_ticks([])
    ax2[1, 2].get_yaxis().set_ticks([])
    ax2[1, 2].set_title("L2-Map3ReLU")

    ax2[2, 0].imshow(l2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax2[2, 0].get_xaxis().set_ticks([])
    ax2[2, 0].get_yaxis().set_ticks([])
    ax2[2, 0].set_title("L2-Map1ReLUPool")

    ax2[2, 1].imshow(l2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
    ax2[2, 1].get_xaxis().set_ticks([])
    ax2[2, 1].get_yaxis().set_ticks([])
    ax2[2, 1].set_title("L2-Map2ReLUPool")

    ax2[2, 2].imshow(l2_feature_map_relu_pool[:, :, 2]).set_cmap("gray")
    ax2[2, 2].get_xaxis().set_ticks([])
    ax2[2, 2].get_yaxis().set_ticks([])
    ax2[2, 2].set_title("L2-Map3ReLUPool")

    plt.savefig("L2.png", bbox_inches="tight")
    plt.close(fig2)
    # Layer 3
    ### 绘制第三层（最后一层）的卷积、ReLU处理、池化后的图，因该层 filter 是一通道，因此每个（conv/ReLU/pooling）结果是一个图
    fig3, ax3 = plt.subplots(nrows=1, ncols=3)
    ax3[0].imshow(l3_feature_map[:, :, 0]).set_cmap("gray")
    ax3[0].get_xaxis().set_ticks([])
    ax3[0].get_yaxis().set_ticks([])
    ax3[0].set_title("L3-Map1")

    ax3[1].imshow(l3_feature_map_relu[:, :, 0]).set_cmap("gray")
    ax3[1].get_xaxis().set_ticks([])
    ax3[1].get_yaxis().set_ticks([])
    ax3[1].set_title("L3-Map1ReLU")

    ax3[2].imshow(l3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
    ax3[2].get_xaxis().set_ticks([])
    ax3[2].get_yaxis().set_ticks([])
    ax3[2].set_title("L3-Map1ReLUPool")

    plt.savefig("L3.png", bbox_inches="tight")
    plt.close(fig3)