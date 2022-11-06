import main

login_name = 'michelziv23'

target_profile = 'cristiano'
# OR
#import sys
#target_profile = sys.argv[1] # pass in target profile as argument

from instaloader import Instaloader, Profile
loader = Instaloader()

# login
try:
    loader.load_session_from_file(login_name)
except FileNotFoundError:
    loader.context.log("Session file does not exist yet - Logging in.")
if not loader.context.is_logged_in:
    loader.interactive_login(login_name)
    loader.save_session_to_file()

profile = Profile.from_username(loader.context, target_profile)
followers = profile.get_followers()

loader.context.log()
loader.context.log('Profile {} has {} followers:'.format(profile.username, profile.followers))
loader.context.log()

for follower in followers:
    loader.context.log(follower.username, flush=True)
'''
By_Likes =sorted(profile.get_posts(), key = lambda post: post.likes, reverse= True)



for post in By_Likes:
    L.download_post(post, PROFILE)
    print(post.likes)
'''

