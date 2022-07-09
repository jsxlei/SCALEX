suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(conos))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))

parser <- ArgumentParser(description='Conos for the integrative analysis of multi-batch single-cell transcriptomic profiles')

parser$add_argument("-i", "--input_path", type="character", help="Path contains RNA data")
parser$add_argument("-o", "--output_path", type="character", default='./', help="Output path")
parser$add_argument("-mf", "--minFeatures", type="integer", default=600, help="Remove cells with less than minFeatures features")
parser$add_argument("-mc", "--minCells", type="integer", default=3, help="Remove features with less than minCells cells")
parser$add_argument("-nt", "--n_top_features", type="integer", default=2000, help="N highly variable features")
args <- parser$parse_args()

message('Reading matrix.mtx and metadata.txt in R...')
data <- readMM(paste(args$input_path, '/matrix.mtx', sep=''))
genes <- read.table(paste(args$input_path, '/genes.txt', sep=''), sep='\t')
metadata <- read.csv(paste(args$input_path, '/metadata.txt', sep=''), sep='\t')
row.names(metadata) <- metadata[,1]
metadata <- metadata[,-1]
metadata$batch = as.character(metadata$batch)
# metadata$celltype = as.character(metadata$celltype)

data <- data.frame(t(data))

colnames(data) <- row.names(metadata) 
row.names(data) <- genes[,1]

adata <- CreateSeuratObject(data, 
                            meta.data = metadata,
                            min.cells = args$minCells, 
                            min.features = args$minFeatures)

batch_ <- unique(metadata$batch)
panel.preprocessed <- list()
for (batch in batch_){
    panel.preprocessed[[as.character(batch)]] <- basicSeuratProc(adata@assays$RNA@counts[,(adata@meta.data$batch == batch)], tsne=FALSE, umap=FALSE, verbose=FALSE)
    panel.preprocessed[[as.character(batch)]] <- RunTSNE(panel.preprocessed[[as.character(batch)]], npcs = 30, verbose=FALSE, check_duplicates=FALSE)
}

con <- Conos$new(panel.preprocessed, n.cores=1)
message('Integrating...')
con$buildGraph(k=30, 
               k.self=20, 
               space='PCA', 
               ncomps=30, 
               n.odgenes=args$n_top_features, 
               matching.method='mNN', 
               metric='angular', 
               score.component.variance=TRUE, 
               verbose=FALSE)

con$findCommunities(method=leiden.community, resolution=1)
con$embedGraph(method="UMAP", 
               min.dist=0.1, 
               spread=1, 
               min.prob.lower=1e-3)

embedding <- data.frame(con$embedding)
colnames(embedding) <- c('Umap1','Umap2')

if (!file.exists(args$output_path)){
    dir.create(file.path(args$output_path),recursive = TRUE)
}

write.table(metadata[rownames(embedding),], paste(args$output_path, "/metadata.txt", sep=''), sep='\t')
write.table(embedding, paste(args$output_path, "/integrated.txt", sep=''), sep='\t')
rm(list = ls())