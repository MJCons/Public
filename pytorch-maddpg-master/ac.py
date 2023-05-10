import torch as th


one_actor = th.load('one_actor.pkl', map_location='cpu')
one_actor_1 = th.load('one_actor_1.pkl', map_location='cpu')
two_critic = th.load('two_critic.pkl', map_location='cpu')
two_critic_1 = th.load('two_critic_1.pkl', map_location='cpu')
output1 = one_actor.FC2.weight.data
output2 = one_actor_1.FC2.weight.data
output3 = two_critic.FC3.bias.data
output4 = two_critic_1.FC3.bias.data

if output1.equal(output2):
    print("11111111111111")
if output3.equal(output4):
    print("2222222222222222")
print(output1)
print("```````````````````````````")
print(output2)
print(output3)
print(output4)
