try:
    from model import PredictFace
except ImportError as err:
    exit(err)


def main():
    """
    Main function.
    """
    model = PredictFace()
    # Create the model
    model.create_model()


if __name__ == '__main__':
    main()
