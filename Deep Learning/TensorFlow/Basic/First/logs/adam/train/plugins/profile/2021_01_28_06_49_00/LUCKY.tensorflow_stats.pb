"�=
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(133333��@933333��@A33333��@I33333��@aW�[G�?iW�[G�?�Unknown�
BHostIDLE"IDLE1�����@A�����@a^E�]E��?i��[�&�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1������q@9������q@A������q@I������q@a�����?i+������?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     �`@9     �`@A     �`@I     �`@a�)�j�n�?iҹ3ҹ3�?�Unknown
^HostGatherV2"GatherV2(133333�[@933333�[@A33333�[@I33333�[@a�V�f��?i-���Yz�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1������U@9������U@A������U@I������U@a@�@�{?i��Xl��?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1�����9P@9�����9P@A�����9P@I�����9P@ak�Y��t?i4u�����?�Unknown
�	Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(133333�E@933333�E@A33333�E@I33333�E@a��Pqy�k?i�����?�Unknown
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����LA@9�����LA@A�����LA@I�����LA@a�Bck�f?iF)b���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1������;@9������;@A������;@I������;@a����a?i��s�k�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff&@@9fffff&@@A33333�8@I33333�8@au�'��|_?i���*.�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(133333�6@933333�6@A33333�6@I33333�6@aǒEr=�\?i���8�<�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(13333336@93333336@A3333336@I3333336@aMM\?i���J�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffff�9@9fffff�9@Afffff�2@Ifffff�2@aX?i"*���V�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1������1@9������1@A������1@I������1@a�*�k�oV?i�b�b�?�Unknown
dHostDataset"Iterator::Model(1�����B@9�����B@A������*@I������*@a�zjf%Q?i�AJ�j�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1333333&@9333333&@A333333&@I333333&@aMML?i8U��q�?�Unknown
VHostSum"Sum_2(1      &@9      &@A      &@I      &@aqu���L?i�΀�x�?�Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@aqu���L?ir�s��?�Unknown�
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1������%@9������%@A������%@I������%@a��q��K?ii�����?�Unknown
ZHostArgMax"ArgMax(1������$@9������$@A������$@I������$@a�:�o�BJ?ix�7�3��?�Unknown
`HostGatherV2"
GatherV2_1(1      !@9      !@A      !@I      !@a�Zk�E?iϞ�Ϟ��?�Unknown
[HostPow"
Adam/Pow_1(1������ @9������ @A������ @I������ @a�j�jE?i�����?�Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1������ @9������ @A������ @I������ @ak%Կz)E?i��G�C��?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff @9ffffff @Affffff @Iffffff @a��=j5�D?i�}��}��?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a��h�`C?if��V��?�Unknown
�HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a�J.g��A?i�vVN̫�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@ag�jVA?i>w�!��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����9P@9�����9P@A������@I������@a�zjf%A?i�_2g��?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�zjf%A?i|��{���?�Unknown
� HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1333333@9333333@A333333@I333333@a@?i������?�Unknown
�!HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aqu���<?iO��1��?�Unknown
�"HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      @9      @A      @I      @aqu���<?i^nr���?�Unknown
X#HostEqual"Equal(1������@9������@A������@I������@ap�.�*�:?i�����?�Unknown
}$HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1333333@9333333@A333333@I333333@a z z8?i9�7��?�Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_3(1������@9������@A������@I������@a��Lmu�7?iҼw&��?�Unknown
b&HostDivNoNan"div_no_nan_1(1������@9������@A������@I������@a��Lmu�7?ikf%��?�Unknown
o'HostReadVariableOp"Adam/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@al���t7?i^�}����?�Unknown
[(HostAddV2"Adam/add(1333333@9333333@A333333@I333333@ak���J�5?i]��[���?�Unknown
v)HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1������@9������@A������@I������@a�j�j5?i�jسj��?�Unknown
e*Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a��=j5�4?if�����?�Unknown�
Y+HostPow"Adam/Pow(1ffffff@9ffffff@Affffff@Iffffff@a��h�`3?i#�2�s��?�Unknown
`,HostDivNoNan"
div_no_nan(1ffffff
@9ffffff
@Affffff
@Iffffff
@a����0?i��4I���?�Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1������	@9������	@A������	@I������	@a���eUQ0?it��s���?�Unknown
�.HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1������	@9������	@A������	@I������	@a���eUQ0?iIM�����?�Unknown
X/HostCast"Cast_3(1������@9������@A������@I������@a���t��/?ix��w���?�Unknown
�0HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a���t��/?i��<Q���?�Unknown
t1HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a����.?i/�>���?�Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @a����.?i��@ai��?�Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1333333@9333333@A333333@I333333@as>�j�-?i�B�B��?�Unknown
w4HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333@9333333@A333333@I333333@as>�j�-?iyƙ���?�Unknown
�5HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@as>�j�-?iZJF���?�Unknown
�6HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1������@9������@A������@I������@a@�@�+?i�H����?�Unknown
~7HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1������@9������@A������@I������@ap�.�*�*?i�U��U��?�Unknown
v8HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a�j�n)?i"CK����?�Unknown
u9HostReadVariableOp"div_no_nan/ReadVariableOp(1333333@9333333@A333333@I333333@a z z(?i��Lmu��?�Unknown
w:HostReadVariableOp"div_no_nan/ReadVariableOp_1(1������@9������@A������@I������@a�*�k�o&?i��j���?�Unknown
�;HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������ @9������ @A������ @I������ @a�j�j%?i�ݤ3��?�Unknown
]<HostCast"Adam/Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��h�`#?i i�i��?�Unknown
v=HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?a�[�["?i؎�׎��?�Unknown
�>HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1333333�?9333333�?A333333�?I333333�?ag�jV!?i�N�>���?�Unknown
T?HostMul"Mul(1�������?9�������?A�������?I�������?a���eUQ ?iT��S���?�Unknown
y@HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a���eUQ ?i�Ui���?�Unknown
aAHostIdentity"Identity(1      �?9      �?A      �?I      �?ajU��e?i      �?�Unknown�*�<
uHostFlushSummaryWriter"FlushSummaryWriter(133333��@933333��@A33333��@I33333��@a�{�ǯ�?i�{�ǯ�?�Unknown�
oHost_FusedMatMul"sequential/dense/Relu(1������q@9������q@A������q@I������q@a]wg;S�?i:�1O���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     �`@9     �`@A     �`@I     �`@a��x�l;�?i�O-��F�?�Unknown
^HostGatherV2"GatherV2(133333�[@933333�[@A33333�[@I33333�[@a�
�%i��?iӟ�Zͱ�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1������U@9������U@A������U@I������U@a����ڄ?i�_98�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1�����9P@9�����9P@A�����9P@I�����9P@a
5�)oT?iL��C�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(133333�E@933333�E@A33333�E@I33333�E@a��jq91u?i�蕊Cn�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����LA@9�����LA@A�����LA@I�����LA@aɢ���p?i�G-`���?�Unknown
�	HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1������;@9������;@A������;@I������;@a�,Ǳ�j?i�t�Q��?�Unknown
�
HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1fffff&@@9fffff&@@A33333�8@I33333�8@ayv~��g?iQ�	�)��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(133333�6@933333�6@A33333�6@I33333�6@a�p���e?iec����?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(13333336@93333336@A3333336@I3333336@a{l�oe?i�Ϝ����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffff�9@9fffff�9@Afffff�2@Ifffff�2@a]�!�`?b?ik�N����?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1������1@9������1@A������1@I������1@ae����`?ii�5��?�Unknown
dHostDataset"Iterator::Model(1�����B@9�����B@A������*@I������*@a������Y?ieП��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1333333&@9333333&@A333333&@I333333&@a{l�oU?i����h(�?�Unknown
VHostSum"Sum_2(1      &@9      &@A      &@I      &@a�=� �=U?i��T3�?�Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a�=� �=U?i�$��=�?�Unknown�
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1������%@9������%@A������%@I������%@a� �c&U?i��1,H�?�Unknown
ZHostArgMax"ArgMax(1������$@9������$@A������$@I������$@a|�����S?i*���R�?�Unknown
`HostGatherV2"
GatherV2_1(1      !@9      !@A      !@I      !@a-Gz��iP?iN�4�RZ�?�Unknown
[HostPow"
Adam/Pow_1(1������ @9������ @A������ @I������ @a
��U8P?iS�.ob�?�Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1������ @9������ @A������ @I������ @a��6�P?i:�yrj�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff @9ffffff @Affffff @Iffffff @a���O?i6]r�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a
Ct�YM?i���y�?�Unknown
�HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a,f�@�K?im���u��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a�q�M�BJ?i�G�~��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�����9P@9�����9P@A������@I������@a������I?i�D�|~��?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a������I?i�A[z���?�Unknown
�HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1333333@9333333@A333333@I333333@a'��TH?i	�����?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a�=� �=E?i����Z��?�Unknown
� HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      @9      @A      @I      @a�=� �=E?i'�&e���?�Unknown
X!HostEqual"Equal(1������@9������@A������@I������@a��ȳ�D?i[�S����?�Unknown
}"HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1333333@9333333@A333333@I333333@a��V͆�B?i9R��?�Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_3(1������@9������@A������@I������@a�l�S�&B?i�'\�۲�?�Unknown
b$HostDivNoNan"div_no_nan_1(1������@9������@A������@I������@a�l�S�&B?iK�Ze��?�Unknown
o%HostReadVariableOp"Adam/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a�����A?iȝ�Mֻ�?�Unknown
[&HostAddV2"Adam/add(1333333@9333333@A333333@I333333@a?�Hm3�@?i�����?�Unknown
v'HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1������@9������@A������@I������@a
��U8@?i���/��?�Unknown
e(Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a���??i�^� ��?�Unknown�
Y)HostPow"Adam/Pow(1ffffff@9ffffff@Affffff@Iffffff@a
Ct�Y=?iX�!ƫ��?�Unknown
`*HostDivNoNan"
div_no_nan(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�}WZ}9?iH�Li���?�Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1������	@9������	@A������	@I������	@aM�g^�8?i�U���?�Unknown
�,HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1������	@9������	@A������	@I������	@aM�g^�8?i��@	��?�Unknown
X-HostCast"Cast_3(1������@9������@A������@I������@a��s��7?i�<Uu��?�Unknown
�.HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a��s��7?iP�é��?�Unknown
t/HostReadVariableOp"Adam/Cast/ReadVariableOp(1      @9      @A      @I      @a�����+7?i���&���?�Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_1(1      @9      @A      @I      @a�����+7?ix�����?�Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1333333@9333333@A333333@I333333@am�s�-f6?ii���?�Unknown
w2HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333@9333333@A333333@I333333@am�s�-f6?idAG/j��?�Unknown
�3HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@am�s�-f6?i����6��?�Unknown
�4HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1������@9������@A������@I������@a�����4?i��K���?�Unknown
~5HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1������@9������@A������@I������@a��ȳ�4?i,I��T��?�Unknown
v6HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @aDۏ�AO3?i'[�Ӿ��?�Unknown
u7HostReadVariableOp"div_no_nan/ReadVariableOp(1333333@9333333@A333333@I333333@a��V͆�2?i���?�Unknown
w8HostReadVariableOp"div_no_nan/ReadVariableOp_1(1������@9������@A������@I������@ae����0?i���/��?�Unknown
�9HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������ @9������ @A������ @I������ @a
��U80?i%Xq�6��?�Unknown
]:HostCast"Adam/Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a
Ct�Y-?ii�rm��?�Unknown
v;HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�������?9�������?A�������?I�������?avZ4J�+?i�?R���?�Unknown
�<HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_2(1333333�?9333333�?A333333�?I333333�?a�q�M�B*?i�[m��?�Unknown
T=HostMul"Mul(1�������?9�������?A�������?I�������?aM�g^�(?i�A����?�Unknown
y>HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?aM�g^�(?ih�'k���?�Unknown
a?HostIdentity"Identity(1      �?9      �?A      �?I      �?a�+� 6�?i      �?�Unknown�2GPU