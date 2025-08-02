# terminalの引数では設定できません。ダイレクトにこのファイルを編集してください。
from hf_rvc.tools.realtime_vcEZ_largeContext import (
    realtime_vc, 
    list_audio_devices, 
    find_audio_device
)

if __name__ == "__main__":
    find_deviceOnly = False  # Set to True if you only want to find devices without starting RVC
    # List available audio devices first
    print("Available Audio Devices:")
    print("=" * 50)
    try:
        print("\n1. All available devices:")
        list_audio_devices()
    except Exception as e:
        print(f"Error listing devices: {e}")

    input = find_audio_device("Your favorite device", "input")
    output = find_audio_device("Your favorite device", "output")
    
    print("\nStarting RVC with sounddevice...")
    print("=" * 50)
    
    if not find_deviceOnly:
        realtime_vc(
            f0_up_key=14.8, # 男声->女声
            expected_audio_level=0.06,  # Expected audio level for dynamic range control
            noise_detection_threshold= 0.001,  # Threshold for noise detection
            input_device_index=input,  # Use the USB input device found
            output_device_index=output,  # Use the Bluetooth output device found
            modelname="Your favorite model",  # Specify the model name or path
        )
