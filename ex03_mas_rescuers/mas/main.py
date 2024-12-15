import sys
import os
import time

# importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

def main(data_folder_name, config_ag_folder_name):
    """
    Configura e executa o simulador de ambiente.
    """
    # Define os caminhos para os arquivos de configuração e dados do ambiente
    current_folder = os.path.abspath(os.getcwd())
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
    
    # Instancia o ambiente
    env = Env(data_folder)
    
    # Instancia o master_rescuer
    # Este agente unifica os mapas e instancia outros 3 agentes
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4)  # 4 é o número de agentes exploradores

    # Cada explorer precisa conhecer o master_rescuer para enviar o mapa
    # Por isso, o master_rescuer é instanciado primeiro
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        Explorer(env, explorer_file, master_rescuer)

    # Executa o simulador de ambiente
    env.run()

if __name__ == '__main__':
    """
    Permite configurar um diretório diferente para os dados usando argumentos de linha de comando.
    Caso contrário, usa os valores padrão.
    """
    if len(sys.argv) > 2:
        # Se dois argumentos são fornecidos, usa-os para data_folder_name e config_ag_folder_name
        data_folder_name = sys.argv[1]
        config_ag_folder_name = sys.argv[2]
    else:
        # Caso contrário, usa valores padrão
        data_folder_name = os.path.join("datasets", "data_300v_90x90")
        config_ag_folder_name = os.path.join("ex03_mas_random_dfs", "cfg_1")
    
    # Executa a função principal com os parâmetros definidos
    main(data_folder_name, config_ag_folder_name)
