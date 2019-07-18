import argparse, requests, time, os

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True)
ap.add_argument('-n', '--num-images', type=int, default=500)
args = vars(ap.parse_args())

url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

for i in range(0, args["num_images"]):
    try:
        r = requests.get(url, timeout=60)

        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        print("[INFO] downloaded: {}".format(p))
        total += 1

    except:
        print("[INFO] error downloading image...")
    
    time.sleep(0.1)
    