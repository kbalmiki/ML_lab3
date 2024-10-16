import turtle
# Ex 14.1
def sed(pattern, replace, source, dest):
    input_file = open(source, 'r')
    output_file = open(dest, 'w')

    for line in input_file:
        line = line.replace(pattern, replace)
        output_file.write(line)

    input_file.close()
    output_file.close()


pattern = 'pattern'
replace = 'replace'
source = 'sed_tester.txt'
dest = source + '.replaced'
sed(pattern, replace, source, dest)

# Ex 15.2
class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self, width=0.0, height=0.0, corner=None):
        self.width = width
        self.height = height
        self.corner = corner if corner is not None else Point()


def draw_rect(t, rect):
    for length in rect.width, rect.height, rect.width, rect.height:
        t.fd(length)
        t.rt(90)


bob = turtle.Turtle()

# draw a rectangle
box = Rectangle()
box.width = 100.0
box.height = 200.0
box.corner = Point()
box.corner.x = 50.0
box.corner.y = 50.0

draw_rect(bob, box)

# wait for the user to close the window
turtle.mainloop()

#Task 17.1
class Kangaroo:

    def __init__(self, name, contents=[]):
        self.name = name
        self.pouch_contents = contents

    def __str__(self):
        t = [self.name + ' has pouch contents:']
        for obj in self.pouch_contents:
            s = '    ' + object.__str__(obj)
            t.append(s)
        return '\n'.join(t)

    def put_in_pouch(self, item):
        self.pouch_contents.append(item)


kanga = Kangaroo('Kanga')
roo = Kangaroo('Roo')
kanga.put_in_pouch('wallet')
kanga.put_in_pouch('car keys')
kanga.put_in_pouch(roo)

print('kanga',kanga)
print('roo', roo)

#Task 18.1
class Deck:
    def __init__(self):
        self.cards = []  # Initialize the deck with a list of card objects
        self.build_deck()

    def build_deck(self):
        # Assume cards are represented as tuples (rank, suit) or any other structure
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [(rank, suit) for suit in suits for rank in ranks]

    def shuffle(self):
        import random
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()

    def deal_hands(self, num_hands, cards_per_hand):
        hands = []
        for _ in range(num_hands):
            hand = Hand()
            for _ in range(cards_per_hand):
                if self.cards:
                    hand.add_card(self.deal_card())
                else:
                    break  # Stop if there are not enough cards
            hands.append(hand)
        return hands


class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def __str__(self):
        return f"Hand with cards: {self.cards}"

deck = Deck()
deck.shuffle()
hands = deck.deal_hands(4, 5)  # Deal 4 hands with 5 cards each

for i, hand in enumerate(hands, 1):
    print(f"Hand {i}: {hand}")
