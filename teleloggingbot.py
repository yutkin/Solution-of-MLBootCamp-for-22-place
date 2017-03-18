import telepot

TOKEN = None
CHAT_ID = None
bot = telepot.Bot(TOKEN)

def sendMsg(msg):
  try:
      bot.sendMessage(CHAT_ID, msg)
  except Exception as e:
      print('Error during sending msg: ', e)

def sendPhoto(photo_path):
  try:
      with open(photo_path, 'rb') as photo:
        bot.sendPhoto(CHAT_ID, photo)
  except Exception as e:
      print('Error during sending photo', e)