from DQN_States.Agent import DQN_States
from Advantage_Actor_Critic.Agent import trainAdvantageActorCritic,testAdvantageActorCritic,plotAdvantageActorCritic
from arguments import get_args
args = get_args()
"""
x=DQN_States('citycopter',train_flag =True)
x.train()
"""
if args.algo =='dqn':

    if args.action=='train':
        model=DQN_States(args.env_name,train_flag =True)
        model.train()
    elif args.action=='test':
        model=DQN_States(args.env_name,train_flag =False)
        model.test()
    elif args.action=='plot':
        model=DQN_States(args.env_name,train_flag =False)
        model.plotLoss()
        model.plotRewards()
    else:
        raise Exception('please provide suitable action')

elif args.algo=='a2c':

    if args.action=='train':
        trainAdvantageActorCritic()
    elif args.action=='test':
        testAdvantageActorCritic()
    elif args.action=='plot':
        plotAdvantageActorCritic()
    else:
        raise Exception('please provide suitable action')

else:
    raise Exception('please provide suitable algorithm type')
    
