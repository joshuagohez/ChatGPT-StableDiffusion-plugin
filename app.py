from flask import Flask, request, jsonify, send_file, Response
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import base64

app = Flask(__name__)

# Stable Diffusion API 
stability_api = client.StabilityInference(
    key='STABLE_DIFFUSION_API_KEY',
    verbose=True,  
    engine="stable-diffusion-xl-beta-v2-2-2"
)

# Generate images
@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()

    prompt = data.get('prompt')  
    seed = data.get('seed', None)  
    steps = data.get('steps', 30)  
    cfg_scale = data.get('cfg_scale', 8.0)  
    width = data.get('width', 512)  
    height = data.get('height', 512)  
    samples = data.get('samples', 1)  

    answers = stability_api.generate(  
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        samples=samples,
        sampler=generation.SAMPLER_K_DPMPP_2M
    )

    # Extract and encode
    generated_images = []
    for res in answers:
        for artifact in res.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                # encodes the binary data of the image using Base64 encoding, then decoded to a UTF-8 string
                encoded_image = base64.b64encode(artifact.binary).decode('utf-8')  
                generated_images.append(encoded_image)

    return jsonify(images=generated_images)

# Plugin logo
@app.route('/logo.png', methods=['GET'])
def plugin_logo():
    return send_file('logo.png', mimetype='image/png')


# Plugin manifest
@app.route('/.well-known/ai-plugin.json', methods=['GET'])
def plugin_manifest():
    host = request.headers['Host']
    with open('./.well-known/ai-plugin.json') as file:
        text = file.read()
        return Response(text, mimetype='text/json')


# OpenAPI specification
@app.route('/openapi.yaml', methods=['GET'])
def openapi_spec():
    host = request.headers['Host']
    with open('./openapi.yaml') as file:
        text = file.read()
        return Response(text, mimetype='text/yaml')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)