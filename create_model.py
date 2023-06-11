import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from parse_json import load_data
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "Athletic Pants",
    "Athletic Sets",
    "Athletic Shirts",
    "Athletic Shorts",
    "Baggy Jeans",
    "Batwing Tops",
    "Beach & Swim Wear",
    "Bikinis",
    "Binders",
    "Blouses",
    "Bodysuits",
    "Boots",
    "Bra Straps",
    "Bubble Coats",
    "Business Shoes",
    "Capes & Capelets",
    "Capri Pants",
    "Cardigans",
    "Cargo Pants",
    "Cargo Shorts",
    "Casual Dresses",
    "Casual Pants",
    "Casual Shirts",
    "Casual Shoes",
    "Casual Shorts",
    "Cleats",
    "Clubbing Dresses",
    "Cocktail Dresses",
    "Corsets",
    "Costumes & Cosplay",
    "Crop Tops",
    "Custom Made Clothing",
    "Dance Wear",
    "Drawstring Pants",
    "Dress Shirts",
    "Dresses",
    "Fashion Sets",
    "Flats",
    "Formal Dresses",
    "Halter Tops",
    "Harem Pants",
    "Heels",
    "Hiking Boots",
    "Hoodies & Sweatshirts",
    "Hosiery, Stockings, Tights",
    "Jackets",
    "Jeans",
    "Jerseys",
    "Jilbaab",
    "Jumpsuits Overalls & Rompers",
    "Kimonos",
    "Leggings",
    "Lingerie Sleepwear & Underwear",
    "Loafers & Slip-on Shoes",
    "Maternity",
    "Nightgowns",
    "Padded Bras",
    "Pajamas",
    "Party Dresses",
    "Pasties",
    "Peacoats",
    "Pencil Skirts",
    "Petticoats",
    "Polos",
    "Prom Dresses",
    "Pullover Sweaters",
    "Rain Boots",
    "Raincoats",
    "Robes",
    "Running Shoes",
    "Sandals",
    "Sheer Tops",
    "Shoe Accessories",
    "Shoe Inserts",
    "Shoelaces",
    "Shorts",
    "Skinny Jeans",
    "Skirts",
    "Slippers",
    "Sneakers",
    "Sports Bras",
    "Stilettos",
    "Suits & Blazers",
    "Sweatpants",
    "Swim Trunks",
    "Swimsuit Cover-ups",
    "Swimsuits",
    "T-Shirts",
    "Tank Tops",
    "Thermal Underwear",
    "Thigh Highs",
    "Thongs",
    "Three Piece Suits",
    "Trench Coats",
    "Trousers",
    "Tube Tops",
    "Tutus",
    "Undershirts",
    "Underwear",
    "Uniforms",
    "Vests",
    "Wedding Dresses",
    "Wedges & Platforms",
    "Winter Boots",
    "Yoga Pants",
]


def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(256, 256)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(28),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    images, labels = load_data()
    print(load_data())
    train_images, train_labels, test_images, test_labels = train_test_split(
        images, labels, test_size=0.33
    )
    model = create_model()
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)
    model.save("data/my_model.h5")
