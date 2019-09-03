try:
    from model import PredictFace
except ImportError as err:
    exit("{}: {}".format(__file__, err))


def main():
    """
    Main function.
    """
    model = PredictFace(verbose=1)
    # Create the model
    model.create_model()
    # Learn
    model.learn()
    # Save this model
    model.save()


if __name__ == '__main__':
    main()
