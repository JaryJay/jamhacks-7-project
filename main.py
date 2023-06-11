import os
from playsound import playsound
import cv2
import tensorflow as tf
import numpy as np
from get_color import get_colors
from PIL import Image

from gpt import get_message
from tts import get_mp3

# Load the .h5 model
model = tf.keras.models.load_model("data/my_model copy.h5")

# Set up the camera feed
cap = cv2.VideoCapture(0)

# Define the labels for the classes (modify as per your model)
labels = [
    "Argyle",
    "Asymmetric",
    "Athletic Pants",
    "Athletic Sets",
    "Athletic Shirts",
    "Athletic Shorts",
    "Backless Dresses",
    "Baggy Jeans",
    "Bandage",
    "Bandeaus",
    "Batwing Tops",
    "Beach & Swim Wear",
    "Beaded",
    "Beige",
    "Bikinis",
    "Binders",
    "Black",
    "Blouses",
    "Blue",
    "Bodycon",
    "Bodysuits",
    "Boots",
    "Bra Straps",
    "Bronze",
    "Brown",
    "Bubble Coats",
    "Business Shoes",
    "Camouflage",
    "Canvas",
    "Capes & Capelets",
    "Capri Pants",
    "Cardigans",
    "Cargo Pants",
    "Cargo Shorts",
    "Cashmere",
    "Casual Dresses",
    "Casual Pants",
    "Casual Shirts",
    "Casual Shoes",
    "Casual Shorts",
    "Chambray",
    "Checkered",
    "Chevron",
    "Chiffon",
    "Clear",
    "Cleats",
    "Clubbing Dresses",
    "Cocktail Dresses",
    "Collared",
    "Corduroy",
    "Corsets",
    "Costumes & Cosplay",
    "Cotton",
    "Criss Cross",
    "Crochet",
    "Crop Tops",
    "Custom Made Clothing",
    "Dance Wear",
    "Denim",
    "Drawstring Pants",
    "Dress Shirts",
    "Dresses",
    "Embroidered",
    "Fashion Sets",
    "Faux Fur",
    "Female",
    "Hoodie",
    "Flats",
    "Fleece",
    "Floral",
    "Formal Dresses",
    "Fringe",
    "Furry",
    "Galaxy",
    "Geometric",
    "Gingham",
    "Gold",
    "Gray",
    "Green",
    "Halter Tops",
    "Harem Pants",
    "Hearts",
    "Heels",
    "Herringbone",
    "Hi-Lo",
    "Hiking Boots",
    "Hollow-Out",
    "Hoodies & Sweatshirts",
    "Hosiery, Stockings, Tights",
    "Houndstooth",
    "Jackets",
    "Jeans",
    "Jerseys",
    "Jilbaab",
    "Jumpsuits Overalls & Rompers",
    "Kimonos",
    "Knit",
    "Lace",
    "Leather",
    "Leggings",
    "Leopard And Cheetah",
    "Linen",
    "Lingerie Sleepwear & Underwear",
    "Loafers & Slip-on Shoes",
    "Long Sleeved",
    "Male",
    "Marbled",
    "Maroon",
    "Maternity",
    "Mesh",
    "Multi Color",
    "Neoprene",
    "Neutral",
    "Nightgowns",
    "Nylon",
    "Off The Shoulder",
    "Orange",
    "Organza",
    "Padded Bras",
    "Paisley",
    "Pajamas",
    "Party Dresses",
    "Pasties",
    "Patent",
    "Peach",
    "Peacoats",
    "Pencil Skirts",
    "Peplum",
    "Petticoats",
    "Pin Stripes",
    "Pink",
    "Plaid",
    "Pleated",
    "Plush",
    "Polka Dot",
    "Polos",
    "Polyester",
    "Printed",
    "Prom Dresses",
    "Puff Sleeves",
    "Pullover Sweaters",
    "Purple",
    "Quilted",
    "Racerback",
    "Rain Boots",
    "Raincoats",
    "Rayon",
    "Red",
    "Reversible",
    "Rhinestone Studded",
    "Ripped",
    "Robes",
    "Round Neck",
    "Ruched",
    "Ruffles",
    "Running Shoes",
    "Sandals",
    "Satin",
    "Sequins",
    "Sheer Tops",
    "Shoe Accessories",
    "Shoe Inserts",
    "Shoelaces",
    "Short Sleeves",
    "Shorts",
    "Shoulder Drapes",
    "Silk",
    "Silver",
    "Skinny Jeans",
    "Skirts",
    "Sleeveless",
    "Slippers",
    "Snakeskin",
    "Sneakers",
    "Spaghetti Straps",
    "Spandex",
    "Sports Bras",
    "Square Necked",
    "Stilettos",
    "Strapless",
    "Stripes",
    "Suede",
    "Suits & Blazers",
    "Summer",
    "Sweatpants",
    "Sweetheart Neckline",
    "Swim Trunks",
    "Swimsuit Cover-ups",
    "Swimsuits",
    "T-Shirts",
    "Taffeta",
    "Tan",
    "Tank Tops",
    "Teal",
    "Thermal Underwear",
    "Thigh Highs",
    "Thongs",
    "Three Piece Suits",
    "Tie Dye",
    "Trench Coats",
    "Trousers",
    "Tube Tops",
    "Tulle",
    "Tunic",
    "Turtlenecks",
    "Tutus",
    "Tweed",
    "Twill",
    "Two-Tone",
    "U-Necks",
    "Undershirts",
    "Underwear",
    "Uniforms",
    "V-Necks",
    "Velour",
    "Velvet",
    "Vests",
    "Vintage Retro",
    "Vinyl",
    "Wedding Dresses",
    "Wedges & Platforms",
    "White",
    "Winter Boots",
    "Wool",
    "Wrap",
    "Yellow",
    "Yoga Pants",
    "Zebra",
]

DELAY = 80
timeUntilNextFetch = 0

while True:
    # Read frame from the camera feed
    ret, frame = cap.read()

    # Pre-process the frame (resize, normalize, etc.)
    resized_frame = cv2.resize(frame, (64, 64))
    preprocessed_frame = np.expand_dims(resized_frame, axis=0)

    # Pass the pre-processed frame through the model for prediction
    predictions = model.predict(preprocessed_frame)

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions[0])
    # print(max(predictions[0]))
    # print(np.maximum(predictions[0]))
    predicted_class_label = labels[predicted_class_index]

    # Display the predicted class label on the frame
    cv2.putText(
        frame,
        predicted_class_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    if timeUntilNextFetch == 0:
        color = get_colors(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        message = get_message("compliment", f"{color} {predicted_class_label}")
        get_mp3(message)
        playsound("output.mp3")
        os.remove("output.mp3")
        timeUntilNextFetch = DELAY

    timeUntilNextFetch -= 1

    # Display the frame with predictions
    cv2.imshow("Live Feed", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Clear the current frame
    frame = None

# Release resources
cap.release()
cv2.destroyAllWindows()
