# ./rl_agent/rl_config.py
import torch

class RLAgentConfig:
    """
    Configuração para o Agente DQN, ajustada para aprendizado profundo e estável (nível de maestria).
    """
    def __init__(self):
        # --- Arquitetura da Rede ---
        # A arquitetura 256 -> 128 é um bom ponto de partida, não precisa mudar por enquanto.
        self.FC1_UNITS = 128
        self.FC2_UNITS = 128
        
        # --- Hiperparâmetros de Aprendizado ---
        
        # Aumenta a memória para reter uma variedade maior de experiências.
        self.BUFFER_SIZE = int(5e5)  # De 1e5 para 500.000
        
        # Aumenta o tamanho do lote para atualizações de gradiente mais estáveis.
        self.BATCH_SIZE = 128        # De 64 para 128
        
        # Fator de desconto padrão. Mantido.
        self.GAMMA = 0.99
        
        # Taxa de atualização suave da rede alvo. Mantida.
        self.TAU = 1e-3
        
        # Diminui a taxa de aprendizagem para um ajuste fino e mais estável da política.
        self.LR = 1e-3               # De 5e-4 para 1e-4 (0.0001)
        
        # Frequência de atualização da rede. Mantida.
        self.UPDATE_EVERY = 4

        # --- Dispositivo ---
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

class TrainingConfig:
    """
    Configuração para o loop de treinamento, ajustada para exploração prolongada e maestria.
    """
    def __init__(self):
        # Aumenta o número de episódios para dar tempo suficiente para o aprendizado convergir.
        self.N_EPISODES = 5000       # De 2000 para 5000 (ou mais, se tiver tempo)
        
        # Máximo de passos por episódio. Mantido.
        self.MAX_T = 5000
        
        # Epsilon inicial. Mantido em 1.0 para começar com exploração total.
        self.EPS_START = 1.0
        
        # Epsilon final. Mantido.
        self.EPS_END = 0.01
        
        # Diminui drasticamente o decaimento do epsilon para uma "infância" de exploração mais longa.
        # Isso é crucial para evitar a convergência prematura para uma estratégia ruim.
        self.EPS_DECAY = 0.999      # De 0.999 para 0.9997
        
        # --- Política de Salvamento ---
        # Aumenta o intervalo para salvar, pois o aprendizado será mais gradual.
        self.SAVE_INTERVAL_EPISODES = 250 # De 100 para 250
        self.SAVE_INTERVAL_SECONDS = 900 # De 15 para 30 minutos

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])