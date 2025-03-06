import samna

try:
    devkit = samna.device.open_device("Speck2fDevKit:0")
    print("Devkit opened successfully:", devkit)
except Exception as e:
    print("Error opening devkit:", e)
Z