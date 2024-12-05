import json


def preprocess_data(input_file, output_file):
   # Reading the JSON array from the input file
   with open(input_file, 'r', encoding='utf-8') as infile:
       data = json.load(infile)  # Load the JSON array


   # Writing the data to the output file in JSON format
   with open(output_file, 'w', encoding='utf-8') as outfile:
       for entry in data:
           context = entry.get("context")
           question = entry.get("question")
           answer = entry.get("answer")
           if context and question and answer:
               json.dump({"context": context, "question": question, "answer": answer}, outfile)
               outfile.write('\n')


# Example usage
preprocess_data("../data/raw_data.json", "../data/processed_data.json")



