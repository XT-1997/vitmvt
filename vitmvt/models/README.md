## Mutables
`Mutables`意为可变的(可搜索的)层，是GML算法中一个基础的可搜索层。在GML的剪枝、NAS、量化算法中，我们往往需要在训练的过程中动态的调整某些层。比如NAS里需要搜索某些层的类型、kernel_size；剪枝算法里需要搜索通道数；量化算法里需要搜索Bits数。

`Mutables`中仅包含必要操作的权重，比如候选操作本身的权重。其它如何调整`Mutables`的属性，比如架构的状态应该放在`Mutator`中。它含有一个`key`属性，用于识别其唯一身份的id。用户可以使用此`key`在网络的各个地方进行共享。默认情况下，会产生一个从1开始的全局唯一的ID。

### MixedOp

### SliceOp

## Mutators

### RandomMutator


## Algorithms

### distiller

### nas

### purning


## SuperNet

### 如何使用现有的supernet

### 如何定义一个新的supernet
