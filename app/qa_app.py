from PyQt5.QtWidgets import (
   QApplication, QWidget, QLabel, QTextEdit, QLineEdit, QPushButton, QVBoxLayout
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import sys


class QAApp(QWidget):
   def __init__(self):
       super().__init__()
       self.setWindowTitle("QA System")


       # Load the fine-tuned model
       model = AutoModelForSeq2SeqLM.from_pretrained("../scripts/models/fine_tuned_llm")
       tokenizer = AutoTokenizer.from_pretrained("../scripts/models/fine_tuned_llm")
       self.qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


       # Create widgets
       self.context_label = QLabel("Context:")
       self.context_text = QTextEdit()
       self.context_text.setPlaceholderText("Enter the context here...")
       self.question_label = QLabel("Question:")
       self.question_entry = QLineEdit()
       self.question_entry.setPlaceholderText("Enter your question here...")
       self.submit_button = QPushButton("Ask")
       self.submit_button.clicked.connect(self.get_answer)
       self.answer_label = QLabel("Answer:")
       self.answer_text = QTextEdit()
       self.answer_text.setReadOnly(True)


       # Layout widgets
       layout = QVBoxLayout()
       layout.addWidget(self.context_label)
       layout.addWidget(self.context_text)
       layout.addWidget(self.question_label)
       layout.addWidget(self.question_entry)
       layout.addWidget(self.submit_button)
       layout.addWidget(self.answer_label)
       layout.addWidget(self.answer_text)
       self.setLayout(layout)


   def get_answer(self):
       context = self.context_text.toPlainText().strip()
       question = self.question_entry.text().strip()


       if context and question:
           input_text = f"question: {question} context: {context}"
           answer = self.qa_pipeline(input_text, max_length=64, do_sample=False)[0]['generated_text']
           self.answer_text.setPlainText(answer)
       else:
           self.answer_text.setPlainText("Please provide both context and question.")


if __name__ == "__main__":
   app = QApplication(sys.argv)
   qa_app = QAApp()
   qa_app.show()
   sys.exit(app.exec_())





