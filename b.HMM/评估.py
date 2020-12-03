# 评估描述
# 给定观测序列O（o1,o2,…,oT）和模型u = (π,A,B),求出P（O | u）

# 定义前向变量算子αt（i）=P(O1,O2,…,Ot,Xt = Si | u),前向变量表示t时刻，si状态结束是之前路径上的总概率
class HMM:
    def __init__(self, ):
        pass

    def _forward(self, observationsSeq):
        T = len(observationsSeq)
        N = len(self.pi)
        alpha = np.zeros((T, N), dtype=float)
        alpha[0, :] = self.pi * self.B[:, observationsSeq[0]]  # numpy可以简化循环
        for t in range(1, T):
            for n in range(0, N):
                alpha[t, n] = np.dot(alpha[t - 1, :], self.A[:, n]) * self.B[n, observationsSeq[t]]  # 使用内积简化代码
        return alpha

if __name__ == '__main__':
    print(1<<100)