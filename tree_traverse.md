```py


tree level 0:
 tree_mask: torch.Size([1, 1, 3, 3]),
 input_hidden: torch.Size([1, 3, 2048]),
 input_ids: [[16, 220, 271]],
 logprobs: [-0.466552734375, -2.576171875, -2.716796875],
 topk_cs_index: [0, 1, 2],
 logprobs_list: [
  tensor([[-0.4666, -2.5762, -2.7168]], device='cuda:0', dtype=torch.float16)],
 parents_list: [
  tensor([0], device='cuda:0')],
 ss_token: [
  tensor([[ 16, 220, 271]], device='cuda:0')]

tree level 1:
 tree_mask: torch.Size([1, 1, 3, 6]),
 input_hidden: torch.Size([1, 3, 2048]),
 input_ids: [[15, 271, 11]],
 logprobs: [-1.298828125, -2.236328125, -2.275390625],
 topk_cs_index: [0, 1, 2],
 logprobs_list: [
  tensor([[-0.4666, -2.5762, -2.7168]], device='cuda:0', dtype=torch.float16),
  tensor([[-1.2988, -2.2363, -2.2754],
          [-3.4082, -4.3438, -4.3828],
          [-3.5488, -4.4844, -4.5234]], device='cuda:0', dtype=torch.float16)],
 parents_list: [
  tensor([0], device='cuda:0'),
  tensor([1, 2, 3], device='cuda:0')],
 ss_token: [
  tensor([[ 16, 220, 271]], device='cuda:0'),
  tensor([[ 15, 271,  11],
          [ 16,  15, 220],
          [220,  16, 271]], device='cuda:0')]

tree level 2:
 tree_mask: torch.Size([1, 1, 3, 9]),
 input_hidden: torch.Size([1, 3, 2048]),
 input_ids: [[271, 11, 220]],
 logprobs: [-2.224609375, -2.537109375, -3.162109375],
 topk_cs_index: [0, 1, 3],
 logprobs_list: [
  tensor([[-0.4666, -2.5762, -2.7168]], device='cuda:0', dtype=torch.float16),
  tensor([[-1.2988, -2.2363, -2.2754],
          [-3.4082, -4.3438, -4.3828],
          [-3.5488, -4.4844, -4.5234]], device='cuda:0', dtype=torch.float16),
  tensor([[-2.2246, -2.5371, -4.2734],
          [-3.1621, -3.4746, -5.2109],
          [-3.2012, -3.5137, -5.2500]], device='cuda:0', dtype=torch.float16)],
 parents_list: [
  tensor([0], device='cuda:0'),
  tensor([1, 2, 3], device='cuda:0'),
  tensor([4, 5, 6], device='cuda:0')],
 ss_token: [
  tensor([[ 16, 220, 271]], device='cuda:0'),
  tensor([[ 15, 271,  11],
          [ 16,  15, 220],
          [220,  16, 271]], device='cuda:0'),
  tensor([[271,  11,  15],
          [220, 271,  15],
          [220,  11,  15]], device='cuda:0')]

tree level 3:
 tree_mask: torch.Size([1, 1, 3, 12]),
 input_hidden: torch.Size([1, 3, 2048]),
 input_ids: [[220, 220, 271]],
 logprobs: [-2.931640625, -3.244140625, -3.45703125],
 topk_cs_index: [0, 3, 1],
 logprobs_list: [
  tensor([[-0.4666, -2.5762, -2.7168]], device='cuda:0', dtype=torch.float16),
  tensor([[-1.2988, -2.2363, -2.2754],
          [-3.4082, -4.3438, -4.3828],
          [-3.5488, -4.4844, -4.5234]], device='cuda:0', dtype=torch.float16),
  tensor([[-2.2246, -2.5371, -4.2734],
          [-3.1621, -3.4746, -5.2109],
          [-3.2012, -3.5137, -5.2500]], device='cuda:0', dtype=torch.float16),
  tensor([[-2.9316, -3.4570, -4.6172],
          [-3.2441, -3.7695, -4.9297],
          [-3.8691, -4.3945, -5.5547]], device='cuda:0', dtype=torch.float16)],
 parents_list: [
  tensor([0], device='cuda:0'),
  tensor([1, 2, 3], device='cuda:0'),
  tensor([4, 5, 6], device='cuda:0'),
  tensor([13, 14, 16], device='cuda:0')],
 ss_token: [tensor([[ 16, 220, 271]], device='cuda:0'),
  tensor([[ 15, 271,  11],
          [ 16,  15, 220],
          [220,  16, 271]], device='cuda:0'),
  tensor([[271,  11, 15],
          [220, 271,  15],
          [220,  11,  15]], device='cuda:0'),
  tensor([[220, 271,  11],
          [220,  11, 271],
          [ 16,  15, 220]], device='cuda:0')]



```
