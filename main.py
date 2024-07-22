from waifumem import WaifuMem, Conversation


def main():
    waifumem = WaifuMem()
    conversation = Conversation()

    conversation.add_message("hey", "User A", 1)
    conversation.add_message("yo", "User B", 2)
    conversation.add_message("how's it going?", "User A", 3)
    conversation.add_message("the craziest thing happened today actually", "User B", 4)
    conversation.add_message("I actually got a gf", "User B", 5)
    conversation.add_message("WHAT", "User A", 6)
    conversation.add_message("THERE'S NO WAY", "User C", 7)
    conversation.add_message("bro spill the beans rn", "User A", 8)
    conversation.add_message("ok so basically in my ESL class", "User B", 9)
    conversation.add_message("there's this really cute russian girl", "User B", 10)
    conversation.add_message("and we somehow ended up talking and playing games together", "User B", 11)
    conversation.add_message(":skull:", "User A", 12)
    conversation.add_message("that doesn't make you boyfriend and girlfriend", "User C", 13)
    conversation.add_message("I asked her out", "User B", 14)
    conversation.add_message("and she agreed", "User B", 15)
    conversation.add_message("unfriending you rn", "User A", 16)
    conversation.add_message("nah bro you ain't my friend anymore", "User C", 17)

    waifumem.remember(conversation)


if __name__ == "__main__":
    main()
