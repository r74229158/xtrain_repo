import argparse

from Utils.configs import conf
from Utils.model_utils import *; from Training.Data.Modules.custom_loader import CustomLoader 
from Training.Trainer.DNNTrainer import simpleDNNTrainer
from XAI_Applications.deletion_AUC import test_methods
from Utils.visualize_pam import compare_attributions


def main():

    parser = argparse.ArgumentParser()
    
    # Required positional argument
    parser.add_argument("-conf", "--config_name", type=str, help="Select a particular configuration (model-data) to run XtrAIn",
                        default='simpleDNNTMnist')
    parser.add_argument("-nt",'--num_test_data', type=int, help="The number of samples for XtrAIn to track their attribution scores",
                        default=64)
    parser.add_argument("-br", '--epoch_break', type=int, help="Breaking epochs and saving results according to the epoch passed." \
    "This is particularly useful when the model is trained for many epochs and the results start to ramp up in " \
    "memory ", 
                        default=5)
    
    parser.add_argument("--save_scores", type=str, help="Whether or not to save attributions and heatmaps of all the attribution scores " \
    "for all epochs, only for the last or none at all (only last attribution saved as .pt).", 
                        choices=['none', 'last', 'all'],default='last')
    
    parser.add_argument("--save_pdf", type=bool, help="True to save heatmaps of attribution scores in pdf format. Else, a simple png" \
    "is saved. Important: to apply, the variable `save_scores` should not be `none`.", 
                        default=False)
    
    parser.add_argument("--test_xai", type=bool, help="Boolean variable. If True, a comparison to baseline attribution methods is" \
    "performed", 
                        default=False)
    
    parser.add_argument("--save_other_scores", type=bool, help="Controls saving attribution scores for other methods, if comparison is " \
    "performed, ", 
                        default=False)
    
    args = parser.parse_args()

    run(conf[args.config_name], args.num_test_data, args.epoch_break, 
        args.save_scores, args.save_pdf, args.test_xai, args.save_other_scores)

    
def run(config, test_data, 
        break_per_epochs=5,
        save_r_scores='last',
        save_pdf=False,
        test_xai_methods=False, 
        save_methods_r=False):
    """
    This method performs the XtrAIn algorithm for a particular configuration. It then compares
    the results with other attribution methods. The model architecture and data are selected according
    to the `config` passed. Then the training process starts and XtrAIn starts updating.
    
    Args:
        config (dict): contains all necessary configs
        test_data (int): number of data from the test set to perform XtrAIn
        test_xai_methods (bool): If True, a attribution scores for other methods is 
            calculated and comparison based on Deletion AUC is held
        break_per_epochs (int): a step of the algorithm for saving intermediate results. This is
            performed only for `save_r_scores='all'`, meaning that results from each epoch are saved
        save_r_scores (str): Options in ['all', 'last', 'none']: which R scores to save (R scores from
            XtrAIn are automatically saved as .pt, this variable relates to intermediate R's saved as
            images).
        save_pdf (bool): Whether to save images in pdf format or png.
        save_methods_r (bool): Whether to save images of attribution scores of other methods (works if 
            test_xai_methods==True).
    """

    ## Configuration and training
    model = get_model_architecture(config['model_name'], 
                                   config['model_layers'] + \
                                  [config["num_classes"]])

    data_loader = CustomLoader(config["dataset"], 
                               True, 
                               config["batch_size"],
                               shuffle_test=True)

    trainer = simpleDNNTrainer(model, config,
                               data_loader, 
                               test_data,
                               break_per_epochs=break_per_epochs,
                               run_simple_acc=False,
                               save_r_scores=save_r_scores,
                               save_pdf=save_pdf)
    
    print("--------------------------------------------------")
    print(f"Starting the calculation of **XtrAIn** for: \
          \n \t - dataset: {data_loader.dataset_name} \
          \n \t - number of samples: {test_data} \
          \n \t - epochs: {config['epochs']}")
    print("--------------------------------------------------")
    _, _, model_pth = trainer.train()

    if test_xai_methods:

        if data_loader.dataset_name == 'pam50':
        
            print("PAM50 cannot be used along with Deletion AUC")
            compare_attributions(trainer.run_num)
            return
        
        print("--------------------------------------------------")
        print("Calculating the Deletion AUC criterion for XtrAIn and other baseline methods")
        print("--------------------------------------------------")

        pth = f"run_{trainer.run_num}"
        test_methods(model_pth.rsplit("/")[-1], 
                     config['model_layers'] + [config["num_classes"]],
                     f"/{config["dataset"]}/{pth}",
                     save_methods_r=save_methods_r, 
                     baseline_val=config['baseline_val'])


if __name__ == "__main__":

    main()
