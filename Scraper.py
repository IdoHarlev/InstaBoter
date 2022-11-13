import instaloader

import main
import time


from instaloader import Instaloader, Profile
import pandas as pd
from instaloader.exceptions import ProfileNotExistsException
import random
import os



#insert username to get all of it's followers
def find_User_followers(USR):
    profile = Profile.from_username(loader.context, USR)
    followers = profile.get_followers()
    loader.context.log()
    loader.context.log('Profile {} has {} followers:'.format(profile.username, profile.followers))
    loader.context.log()

    follow_list = []
    count = 0
    for followee in followers:
        follow_list.append(followee.username)
        file = open("userlist", "a+")
        file.write(follow_list[count])
        file.write("\n")
        file.close()
        print(follow_list[count])
        count = count + 1

#Insert Instagram Username to get All Posts and Likes for Each Username
def Create_User_Folder(name):
        counter = 0
        try:
            print("stage 2")
            time.sleep(2)
            profile = Profile.from_username(loader.context, name)

            print("stage 3")

        except ProfileNotExistsException:
            return print("No User Found")
        path = '/Users/ido/Desktop/CoachAI/InstaBoter/' + profile.username
        if not os.path.isdir(path):
            os.makedirs(path)
        os.chdir(path)
        for post in profile.get_posts():
            print("stage 4")
            time.sleep(8)
            if not post.is_video:
                # if post.likes >=0:
                #instaloader.RateController.handle_429(loader.download_post(post,target = str(round(post.likes/profile.followers,3))))
                loader.download_post(post,target = str(post.likes))
                counter +=1
                print(counter)


                print("stage 5")



#login_name = 'michelziv23'
target_profile = input("Enter your User Name: ")
list_of_user_agents = ["agent1", "agent2", "agent3"]
loader = instaloader.Instaloader(sleep = False, max_connection_attempts= 1, user_agent = random.choice(list_of_user_agents ), download_videos= False,save_metadata=False)
loader.save_metadata = False

print(main.Password)
loader.login(main.Username,main.Password)       # (login)
#loader.interactive_login(main.Username)      # (ask password on terminal)
loader.load_session_from_file(main.Username) # (load session created w/

load_users = False




Create_User_Folder(target_profile)










'''
df = pd.read_csv('/Users/ido/Desktop/CoachAI/InstaBoter/userlist', names = ['UserName'])

print("to here")

for i in df['UserName']:

    print("stage 1")

    #instaloader.RateController.handle_429()
    try:
        print("stage 2")




        profile = Profile.from_username(loader.context , i)

        print("stage 3")
        #time.sleep(5)


    except ProfileNotExistsException:

        continue
    for post in profile.get_posts():
        print("stage 4")
        time.sleep(2)
        if not post.is_video:
            #if post.likes >=0:
                print(post.likes)
                #instaloader.RateController.handle_429(loader.download_post(post,target = str(round(post.likes/profile.followers,3))))
                loader.download_post(post,target = profile.username)

                print("stage 5")


'''


