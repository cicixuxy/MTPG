import torch
from itertools import product
from longclip import longclip
from MTPG import MTPG
from img_loader import PilCloudLoader,PilCloudLoader_pre

levels = ['bad', 'poor', 'fair', 'good', 'perfect']

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
aes_prompt = torch.cat(
    [longclip.tokenize(f"An image with {a} aesthetics.")
     for a in product(levels)
    ]
).to(device)

if __name__ == "__main__":
    model = MTPG(input_dim=768)
    model=model.to(device)
    model.load_state_dict(torch.load('checkpoints/model_best.pt'))
    model.eval()
    img_loader = PilCloudLoader(handle_exceptions=False, size=(224,224), aug_num=1)
    image_path = 'images/Img1.jpg'
    input1 = img_loader(image_path,if_train=False).unsqueeze(0).to(device)
    description1=[(f"The painting is executed in a vibrant and expressive style, characterized by bold, thick brushstrokes that create a sense of movement and energy. "
                  f"The composition is dominated by a central figure, a police dog, rendered in a dynamic pose with its head held high and its body lunging forward. "
                  f"The dog's sharp teeth and bared teeth convey a sense of aggression and determination. "
                  f"The background is a blur of colors and shapes, suggesting the chaos and urgency of the scene."
                  f"The use of contrasting colors, such as the bright red of the police car and the deep blue of the night sky, adds to the dramatic effect of the painting.")]
    info_token = longclip.tokenize(description1) .to(device)
    logits_aesthetic = model(input1, info_token, aes_prompt)
    pred1 = 0.1 * logits_aesthetic[:, 0] + 0.3 * logits_aesthetic[:, 1] + 0.5 * logits_aesthetic[:,
                                                                          2] + 0.7 * logits_aesthetic[:,
                                                                                   3] + 0.9 * logits_aesthetic[
                                                                                            :,
                                                                                            4]
    print("The score of Img1 is: %.2f" % pred1.item())
    image_path = 'images/Img2.jpg'
    input2 = img_loader(image_path, if_train=False).unsqueeze(0).to(device)
    description2 = [(f"The painting exhibits a highly realistic and detailed style, capturing every nuance of the subject's features. "
                     f"The artist employs a combination of soft, blended brushstrokes and sharp, precise lines to create a sense of depth and texture. "
                     f"The subject's eyes are rendered with particular attention to detail, conveying a sense of intelligence and introspection. "
                     f"The overall composition is balanced and harmonious, with the subject's gaze drawing the viewer's attention to the center of the canvas. "
                     f"The use of light and shadow adds a dramatic effect, highlighting the subject's facial features and creating a sense of three-dimensionality.")]
    info_token = longclip.tokenize(description2).to(device)
    logits_aesthetic = model(input2, info_token, aes_prompt)
    pred2 = 0.1 * logits_aesthetic[:, 0] + 0.3 * logits_aesthetic[:, 1] + 0.5 * logits_aesthetic[:,
                                                                                2] + 0.7 * logits_aesthetic[:,
                                                                                           3] + 0.9 * logits_aesthetic[
                                                                                                      :,
                                                                                                      4]
    print("The score of Img2 is: %.2f" % pred2.item())

