from DQN_States.Agent import DQN_States
from DQN_Screenshots.Agent_DQN_ScreenShots import train_DQN_Screenshots,test_DQN_Screenshots,plot_DQN_Screenshots
#from Advantage_Actor_Critic.Agent import trainAdvantageActorCritic,testAdvantageActorCritic,plotAdvantageActorCritic
from arguments import get_args
args = get_args()
"""
x=DQN_States('citycopter',train_flag =True)
x.train()
"""
if args.algo =='dqn':
    if args.train_type =='states':
        print('states')
        if args.action=='train':
            model=DQN_States(args.env_name,train_flag =True,folder=args.folder)
            model.train()
        elif args.action=='test':
            model=DQN_States(args.env_name,train_flag =False,folder=args.folder)
            model.test()
        elif args.action=='plot':
            model=DQN_States(args.env_name,train_flag =False,folder=args.folder)
            model.plotLoss()
            model.plotRewards()
        else:
            raise Exception('please provide suitable action')
    elif args.train_type =='screenshots':
        print('screenshots')
        if args.action=='train':
            train_DQN_Screenshots()
        elif args.action=='test':
            test_DQN_Screenshots()
        elif args.action=='plot':
            plot_DQN_Screenshots()
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
    
