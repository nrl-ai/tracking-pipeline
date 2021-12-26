import youtube_dl
import tqdm
f=open("link.txt","r").read().split("\n")
for path in tqdm.tqdm(f):
 try:
  ydl_opts = {}
  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
     ydl.download([path])
 except:
  pass
