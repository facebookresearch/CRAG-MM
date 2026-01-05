# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class CRAGMultiModalPrompts():
    instructions: str
    in_context_examples: str

    def __init__(self) -> None:
        self.instructions: str = INSTRUCTIONS
        self.in_context_examples: str = IN_CONTEXT_EXAMPLES
    
    def get_instructions(self) -> str:
        return self.instructions

    def get_in_context_examples(self) -> str:
        return self.in_context_examples


INSTRUCTIONS: str = """You will be given a question, a ground truth answer, and a model prediction. Your task is to judge if the prediction is correct or not based on the ground truth answer.

## Instructions
Read the question, ground truth answer, and model prediction carefully. Follow the step by step guideline below to make a judgment. 
1. If the prediction indicates uncertainty or refusal to answer, output "Result: WRONG".
2. If the prediction exactly matches the ground truth, output "Result: CORRECT".
3. If the ground truth is a number
    3.1 If the prediction gives a number that almost exactly matches the ground truth, output "Result: CORRECT". 
    3.2 If the prediction gives a number that is not the same as the ground truth, output "Result: WRONG".
4. If the prediction is self-contradictory, output "Result: WRONG".
5. If the prediction is not answering the question, output "Result: WRONG".
6. If ground truth contains a set of objects, 
    6.1 if the prediction contains exactly same objects as the ground truth, output "Result: CORRECT".
    6.2 if the prediction contains different objects from the ground truth, output "Result: WRONG".
    6.3 if the prediction is almost same as the ground truth, use your best judgement to give output.
7. If the prediction is grounded by the ground truth, output "Result: CORRECT".
8. If the prediction is unrelated or contradictory to the ground truth, output "Result: WRONG".


## Additional Guidelines
- Take it as granted that the ground truth is always correct.
- If the prediction gives extra information that is not in the ground truth, it is still correct as long as it is grounded by the ground truth.
- Be careful about numbers. 1 mile is about 1.60934 km. 1 foot is about 0.3048 m. 1 inch is about 2.54 cm. 1 yard is about 0.9144 m. 1 pound is about 0.453592 kg. 1 gallon is about 3.78541 liters. 1 ounce is about 28.3495 grams.

## Output Format
Your judgment should first provide a VERY-SHORT explanation on your rationale. When relevant, you need to include the guidelines above to explain your judgment. Finally, your judgment should clearly state "Result: CORRECT" or "Result: WRONG".
"""

IN_CONTEXT_EXAMPLES: str = """## Examples
Below are some examples:

EXAMPLES START
Question: who will win the game?
Ground Truth: Lakers is favored to win the game.
Prediction: Sorry, it is hard to predict the outcome of the game.
Explanation: The prediction indicates it is not sure about the answer. So the prediction is incorrect according to the guideline 1.
Result: WRONG

Question: what building is this?
Ground Truth: This is the Empire State Building.
Prediction: Sorry, I cannot help with that.
Explanation: The prediction refuses to answer. So the prediction is incorrect according to the guideline 1.
Result: WRONG

Question: who authored this?
Ground Truth: William Shakespeare authored this book.
Prediction: william shakespeare authored this.
Explanation: The prediction exactly matches the ground truth. So the prediction is correct according to the guideline 2.
Result: CORRECT

Question: how deep is the deepest lake of new york?
Ground Truth: It is about 618 ft.
Prediction: The deepest lake in new york is seneca lake, with a depth of 618.23 feet.
Explanation: The number of the depth in the prediction is almost the same as the ground truth. So the prediction is correct according to the guideline 3.1.
Result: CORRECT

Question: what is the height of this building?
Ground Truth: The height of the building where citigroup is headquartered is 151 m.
Prediction: The height is 915 feet (279 m).
Explanation: The prediction, 151 m, does not match the ground truth, 279 m. So the prediction is incorrect according to the guideline 3.2.
Result: WRONG

Question: what is the current market cap of this company?
Ground Truth: The current market cap of Apple is 2.81 trillion.
Prediction: It is about 2.667 trillion.
Explanation: The number of the market cap in the prediction does not match the ground truth. So the prediction is incorrect according to the guideline 3.2.
Result: WRONG

Question: what is the population of this country?
Ground Truth: The population is 3,576,873.
Prediction: The population of this country is 3.3 million.
Explanation: The number of the population in the prediction is 3.3 million, which is not the same as the 3.58 million in the ground truth. So the prediction is incorrect according to the guideline 3.2.
Result: WRONG

Question: who is older, James or John?
Ground Truth: James is older than John.
Prediction: James is 12 years old and John is 13 years old, and James is older than John.
Explanation: The prediction is self-contradictory as it says James is 12 years old and John is 13 years old, but also says James is older than John. So the prediction is incorrect according to the guideline 4.
Result: WRONG

Question: how old is michael jordan?
Ground Truth: Michael Jordan is 61 years old.
Prediction: MJ was born on February 17, 1963.
Explanation: The prediction does not give the age of MJ. So the prediction is incorrect according to the guideline 5.
Result: WRONG

Question: who are current band members of this band?
Ground Truth: The members of Maroon 5 are adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar.
Prediction: They are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine.
Explanation: The prediction contains the same members as the ground truth. So the prediction is correct according to the guideline 6.1.
Result: CORRECT

Question: what flavors are listed on the sign on the right?	
Ground Truth: the sign on the right lists mango and strawberry flavors.	
Prediction: The sign on the right lists the following flavors: * Orange * Apple
Explanation: The prediction does not contain the same flavors as the ground truth. So the prediction is incorrect according to the guideline 6.2.
Result: WRONG

Question: who authored this?
Ground Truth: William Shakespeare authored Taming of the Shrew.
Prediction: w shakespeare.
Explanation: The prediction is grounded by the ground truth as W is likely the abbreviation of William. So the prediction is correct according to the guideline 7.
Result: CORRECT

Question: who is taller, a or b?
Ground Truth: a is taller than b.
Prediction: a.
Explanation: The prediction is grounded by the ground truth. So the prediction is correct according to the guideline 7.
Result: CORRECT

Question: what's the name of person in the picture?
Ground Truth: The name of the person is lana clarkson.
Prediction: The name is lana clarkson, who is born april 5, 1962.
Explanation: The prediction is grounded by the ground truth. So the prediction is correct according to the guideline 7.
Result: CORRECT

Question: what is the this bird called?
Ground Truth: This bird is called California Quail.
Prediction: California Valley Quail.
Explanation: The prediction is grounded by the ground truth as California Valley Quail is also known as California Quail. So the prediction is correct according to the guideline 7.
Result: CORRECT

Question: is this edible?
Ground Truth: yes, the torch ginger plant is an important ingredient in food across southeast asia.
Prediction: yes, the flower is edible.
Explanation: The prediction is grounded by the ground truth as the flower is edible. So the prediction is correct according to the guideline 7.
Result: CORRECT

Question: what country is this company located in?
Ground Truth: The Maharashtra Metro Rail Corporation Limited is located in India.
Prediction: This Corporation is located in Indianapolis.
Explanation: The prediction talks about Indianapolis, which is not in India. So the prediction is incorrect according to the guideline 8.
Result: WRONG

Question: what kind of animal is this?
Ground Truth: this is a lion-dog, also known as a komainu.
Prediction: The animal depicted in the image is a lion.
Explanation: The prediction says the animal is a lion, which is contradictory to the lion-dog in the ground truth. So the prediction is incorrect according to the guideline 8.
Result: WRONG

Question: which side of these equations is solved first?
Ground Truth: calculate the numbers inside the parenthesis first than multiply the answer with the outside number.
Prediction: The left side of the equations is solved first.
Explanation: The prediction is unrelated to the ground truth. So the prediction is incorrect according to the guideline 8.
Result: WRONG

Question: is this car cheaper than rav4?
Ground Truth: No, the Honda CR-V is more expensive than the Toyota RAV4.
Prediction: no, the nissan frontier is more expensive than the toyota rav4.
Explanation: The prediction talks about the Nissan Frontier, which is unrelated to the Honda CR-V in the ground truth. So the prediction is incorrect according to the guideline 8.
Result: WRONG
EXAMPLES END
"""

