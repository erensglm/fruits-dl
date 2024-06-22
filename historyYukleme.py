
import codecs
import matplotlib.pyplot as plt
import json
with codecs.open("transfer.json","r",encoding="utf-8") as f:
    hist=json.loads(f.read())

plt.plot(hist["loss"],label="Train Loss")
plt.plot(hist["val_loss"],label="Validation Loss")
plt.title("Loss")
plt.legend()
plt.show()
plt.figure()

plt.plot(hist["accuracy"],label="Train Accuracy")
plt.plot(hist["val_accuracy"],label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()
plt.show()
plt.figure()


