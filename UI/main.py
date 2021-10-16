# Importing Libraries
import tkinter as tk
import predictor
import trainer

try:
    import nltk
except:
    import pip
    pip.main(['install', 'nltk'])
    import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Defining functions
def predict():
    global result_label_window, confirmation_buttons_window, prediction
    # Extract message in entry box
    msg = message_entry.get()

    # Get prediction ('Spam' or 'Ham')
    prediction = predictor.predict_spam(msg)

    # Update results
    result_label['text'] = 'Spam' if prediction == 'spam' else 'Not Spam'

    # Update Canvas
    canvas.delete(message_entry_window)
    canvas.delete(pred_button_window)

    result_label_window = canvas.create_window(200, 80, window=result_label)
    confirmation_buttons_window = canvas.create_window(200, 140, window=confirmation_buttons)


def correct_prediction():
    # Extract message and prediction
    msg = message_entry.get()
    label = prediction

    # Update dataset and retrain model
    trainer.update_model(msg, label, correct=True)
    print('Model Updated')

    # Reset Canvass
    message_entry.delete(0, 'end')
    reset_canvas()


def wrong_prediction():
    # Extract message and prediction
    msg = message_entry.get()
    label = prediction
    print('False Label', msg, label)

    # Update dataset and retrain model
    trainer.update_model(msg, label, correct=False)
    print('Model Updated')

    # Reset Canvass
    message_entry.delete(0, 'end')
    reset_canvas()


def next_message():
    # Reset Canvass
    message_entry.delete(0, 'end')
    reset_canvas()


def reset_canvas():
    # Delete current buttons and replace with original scene
    canvas.delete(confirmation_buttons_window)
    canvas.delete(result_label_window)

    global message_entry_window, pred_button_window
    message_entry_window = canvas.create_window(200, 80, window=message_entry)
    pred_button_window = canvas.create_window(200, 140, window=pred_button)


# Initialise window
window = tk.Tk()
window.title("Message Spam Detection")

# Create a canvas
canvas = tk.Canvas(window, width=400, height=200)
canvas.pack()

# Create a empty field box to input user's message
message_entry = tk.Entry(window)
message_entry_window = canvas.create_window(200, 80, window=message_entry)

# Create a button that will predict whether the input message is spam
pred_button = tk.Button(window, text='Filter Spam', relief=tk.RAISED, bd=2, cursor='hand2', command=predict)
pred_button_window = canvas.create_window(200, 120, window=pred_button)

# Label to display the prediction
result_label = tk.Label(window)

# Set of buttons once prediction button is pressed
confirmation_buttons = tk.Frame(window, relief=tk.RAISED)
correct_prediction = tk.Button(confirmation_buttons, text="Correct", font="Courier", cursor='hand2', command=correct_prediction)
wrong_prediction = tk.Button(confirmation_buttons, text="Wrong", font="Courier", cursor='hand2', command=wrong_prediction)
next_prediction = tk.Button(confirmation_buttons, text="Next Message", font="Courier", cursor='hand2', command=next_message)

correct_prediction.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
wrong_prediction.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
next_prediction.grid(row=1, columnspan=2, sticky="ew", padx=5, pady=5)


if __name__ == '__main__':
    window.mainloop()