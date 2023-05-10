import torch as th


one_actor = th.load('eight_actor.pkl', map_location='cpu')
one_actor_1 = th.load('seven_actor.pkl', map_location='cpu')
critic = th.load('critic.pkl', map_location='cpu')
critic_1 = th.load('critic_1.pkl', map_location='cpu')
output1 = one_actor.FC3.weight.data
output2 = one_actor_1.FC3.weight.data
output3 = critic.FC4.bias.data
output4 = critic_1.FC4.bias.data
if output1.equal(output2):
    print("1111111111")
if output3.equal(output4):
    print("2222222222")
print(output1)
print("```````````````````````````")
print(output2)
print(output3)
print(output4)
