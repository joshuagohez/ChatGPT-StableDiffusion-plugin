import requests
import json
import base64

url = 'http://127.0.0.1:5000/generate-image'

# Set up the request payload
payload = {
    'prompt': 'a donut',
    'seed': 992446758,
    'steps': 30,
    'cfg_scale': 8.0,
    'width': 512,
    'height': 512,
    'samples': 1
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    generated_images = data['images']

    for i, encoded_image in enumerate(generated_images):
        # Decode base64-encoded image
        decoded_image = base64.b64decode(encoded_image)

        image_filename = f'generated_image_{i + 1}.png'
        with open(image_filename, 'wb') as image_file:
            image_file.write(decoded_image)

        print(f'Saved generated image {i+1} as {image_filename}')
else:
    print(f'Request failed with status code {response.status_code}')