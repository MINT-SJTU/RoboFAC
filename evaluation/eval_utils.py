import base64
import cv2
import asyncio
import httpx

SEMAPHORE_SIZE = 30

timeout = httpx.Timeout(
    connect=10.0,
    read=200.0,
    write=30.0,
    pool=10.0
)

# The frame rate of our video is 30Hz.
# The default setting is selecting a key frame every 1s.
def video_to_api_input(video_path, prompt, frame_interval=30):
    content_list = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    count = 0
    frame_id = 0
    success, frame = cap.read()

    while success:
        if count % frame_interval == 0:
            # resize to 512x512
            frame = cv2.resize(frame, (512, 512))
            _, buffer = cv2.imencode(".jpg", frame)
            b64_img = base64.b64encode(buffer).decode("utf-8")
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "low"
                }
            })
            frame_id += 1
        count += 1
        success, frame = cap.read()
    print(f"number of frames: {frame_id}")
    cap.release()

    content_list.append({
        "type": "text",
        "text": prompt
    })

    return content_list

async def async_api_request(client, prompt, frame_interval=30, video_path=None, model_name=None, semaphore=None, url=None):
    if video_path is None:
        input_list = prompt
    else:
        input_list = video_to_api_input(video_path, prompt, frame_interval)

    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages":[
            {
                "role": "user",
                "content": input_list
            }
        ],
        "response_format": {
            "type": 'text'
        }
    }
    
    while True:
        try:
            async with semaphore:
                response = await client.post(f"{url}/chat/completions", headers = headers, json = payload, timeout=10000)
                response_json = response.json()
                print(response_json)
                res = response_json["choices"][0]["message"]["content"]
                return res
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)
    
async def async_multi_request(prompts, frame_interval=30, video_paths=None, model_name=None, url=None):
    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    if video_paths is None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [async_api_request(client, prompt, model_name=model_name, semaphore=semaphore, url=url) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            return results
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [async_api_request(client, prompt, frame_interval=frame_interval, video_path=video_path, model_name=model_name, semaphore=semaphore, url=url) for prompt, video_path in zip(prompts, video_paths)]
            results = await asyncio.gather(*tasks)
            return results

def make_eval_prompt(question, pred, ref):
    return f"""You are an expert evaluator. Assess the quality of a model's response to the user's query.

        Question: {question}

        Reference answer: {ref}

        Model's response: {pred}

        Evaluate the model's response on the following criteria: 
        - correctness: factual accuracy and consistency with the reference answer.
        - relevance: how well the model's response addresses the question.
        - completeness: whether all key aspects of the reference answer are covered.
    
        For each criterion, provide a score from 0 to 5 and a **brief** explanation, the score should be an integer.
        The score you give needs to be strict and demanding.

        Output ONLY the JSON object in the following format:
        {{
        "criteria": {{
            "correctness": {{"score": <0-5>, "explanation": <brief explanation>}},
            "relevance": {{"score": <0-5>, "explanation": <brief explanation>}},
            "completeness": {{"score": <0-5>, "explanation": <brief explanation>}},
        }}
        }}
        """