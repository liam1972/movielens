##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)#, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Add genre columns
edx_genre<-edx%>%mutate(Children=as.factor(ifelse(str_detect(genres,"Children")==TRUE,1,0)),
                        Comedy=as.factor(ifelse(str_detect(genres,"Comedy")==TRUE,1,0)),
                        Romance=as.factor(ifelse(str_detect(genres,"Romance")==TRUE,1,0)),
                        Action=as.factor(ifelse(str_detect(genres,"Action")==TRUE,1,0)),
                        Crime=as.factor(ifelse(str_detect(genres,"Crime")==TRUE,1,0)),
                        Thriller=as.factor(ifelse(str_detect(genres,"Thriller")==TRUE,1,0)),
                        Sci_Fi=as.factor(ifelse(str_detect(genres,"Sci-Fi")==TRUE,1,0)),
                        Adventure=as.factor(ifelse(str_detect(genres,"Adventure")==TRUE,1,0)),
                        Drama=as.factor(ifelse(str_detect(genres,"Drama")==TRUE,1,0)),
                        Fantasy=as.factor(ifelse(str_detect(genres,"Fantasy")==TRUE,1,0)),
                        War=as.factor(ifelse(str_detect(genres,"War")==TRUE,1,0)),
                        Western=as.factor(ifelse(str_detect(genres,"Western")==TRUE,1,0)),
                        Mystery=as.factor(ifelse(str_detect(genres,"Mystery")==TRUE,1,0)),
                        Musical=as.factor(ifelse(str_detect(genres,"Musical")==TRUE,1,0)),
                        Animation=as.factor(ifelse(str_detect(genres,"Animation")==TRUE,1,0)),
                        Documentary=as.factor(ifelse(str_detect(genres,"Children")==TRUE,1,0)),
                        Film_noir=as.factor(ifelse(str_detect(genres,"Film-Noir")==TRUE,1,0)),
                        IMAX=as.factor(ifelse(str_detect(genres,"IMAX")==TRUE,1,0)),
                        Horror=as.factor(ifelse(str_detect(genres,"Horror")==TRUE,1,0)))%>%
  select(-genres,-title)
                        
head(edx_genre)
sum(is.na(edx_genre))


#Split edx into train and test set
test_index<-createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train<-edx_genre[-test_index]
edx_test<-edx_genre[test_index]

nrow(edx_train)
nrow(edx_test)

#Do not include users and movies not in the train set
edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

#calculate mean, movie effect and user effect
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
mu <- mean(edx_train$rating)
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, edx_test$rating))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

#Generate regularised movie and user effects with optimised lambda
l<-lambdas[which.min(rmses)]
mu <- mean(edx_train$rating)

b_i <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- edx_train %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

predicted_ratings <- 
  edx_test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8654673

#visualise genre effect using boxplots
#and residual rating after removal of movie and user effects

library(dplyr)
library(ggplot2)

edx_genre_check<-edx_train%>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(resid = rating-(mu + b_i + b_u))
  

p1<-edx_genre_check%>%
  ggplot(aes(Children,resid))+geom_boxplot()
p2<-edx_genre_check%>%
  ggplot(aes(Comedy,resid))+geom_boxplot()
p3<-edx_genre_check%>%
  ggplot(aes(Romance,resid))+geom_boxplot()
p4<-edx_genre_check%>%
  ggplot(aes(Action,resid))+geom_boxplot()
p5<-edx_genre_check%>%
  ggplot(aes(Crime,resid))+geom_boxplot()
p6<-edx_genre_check%>%
  ggplot(aes(Thriller,resid))+geom_boxplot()
p7<-edx_genre_check%>%
  ggplot(aes(Sci_Fi,resid))+geom_boxplot()
p8<-edx_genre_check%>%
  ggplot(aes(Adventure,resid))+geom_boxplot()
p9<-edx_genre_check%>%
  ggplot(aes(Drama,resid))+geom_boxplot()
p10<-edx_genre_check%>%
  ggplot(aes(Fantasy,resid))+geom_boxplot()
p11<-edx_genre_check%>%
  ggplot(aes(War,resid))+geom_boxplot()
p12<-edx_genre_check%>%
  ggplot(aes(Western,resid))+geom_boxplot()
p13<-edx_genre_check%>%
  ggplot(aes(Mystery,resid))+geom_boxplot()
p14<-edx_genre_check%>%
  ggplot(aes(Musical,resid))+geom_boxplot()
p15<-edx_genre_check%>%
  ggplot(aes(Animation,resid))+geom_boxplot()
p16<-edx_genre_check%>%
  ggplot(aes(Documentary,resid))+geom_boxplot()
p17<-edx_genre_check%>%
  ggplot(aes(Film_noir,resid))+geom_boxplot()
p18<-edx_genre_check%>%
  ggplot(aes(IMAX,resid))+geom_boxplot()
p19<-edx_genre_check%>%
  ggplot(aes(Horror,resid))+geom_boxplot()

install.packages("gridExtra")
library(gridExtra)
grid.arrange(p1,p2,p3,p4,p5,p6)

l<-30 # for preliminary assessment

#Regularised Children effect lambda = 30
children<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Children,userId)%>%
  summarize(children=sum(rating-b_i-b_u-mu)/(n()+l))
  
children<-children%>%spread(Children,children)
colnames(children)<-c("userId","children_down","children_up")
children[is.na(children)]<-0
sum(is.na(children$children_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8650211 include

#Comedy
comedy<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Comedy,userId)%>%
  summarize(comedy=sum(rating-b_i-b_u-mu)/(n()+l))

comedy<-comedy%>%spread(Comedy,comedy)
colnames(comedy)<-c("userId","comedy_down","comedy_up")
comedy[is.na(comedy)]<-0
sum(is.na(comedy$comedy_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8632204 include

#Romance
romance<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Romance,userId)%>%
  summarize(romance=sum(rating-b_i-b_u-mu)/(n()+l))

romance<-romance%>%spread(Romance,romance)
colnames(romance)<-c("userId","romance_down","romance_up")
romance[is.na(romance)]<-0
sum(is.na(romance$romance_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(romance,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+ifelse(Romance==1,(romance_up-romance_down)/2,
                                                     -(romance_up-romance_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8640334 don't include

#Action
action<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Action,userId)%>%
  summarize(action=sum(rating-b_i-b_u-mu)/(n()+l))

action<-action%>%spread(Action,action)
colnames(action)<-c("userId","action_down","action_up")
action[is.na(action)]<-0
sum(is.na(action$action_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)
           +ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8601234 include

#Crime
crime<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Crime,userId)%>%
  summarize(crime=sum(rating-b_i-b_u-mu)/(n()+l))

crime<-crime%>%spread(Crime,crime)
colnames(crime)<-c("userId","crime_down","crime_up")
crime[is.na(crime)]<-0
sum(is.na(crime$crime_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           +ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8592462 include

#Thriller
thriller<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Thriller,userId)%>%
  summarize(thriller=sum(rating-b_i-b_u-mu)/(n()+l))

thriller<-thriller%>%spread(Thriller,thriller)
colnames(thriller)<-c("userId","thriller_down","thriller_up")
thriller[is.na(thriller)]<-0
sum(is.na(thriller$thriller_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(thriller,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Thriller==1,(thriller_up-thriller_down)/2,
                  -(thriller_up-thriller_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8608326 don't include

#Sci-Fi
sci_fi<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Sci_Fi,userId)%>%
  summarize(sci_fi=sum(rating-b_i-b_u-mu)/(n()+l))

sci_fi<-sci_fi%>%spread(Sci_Fi,sci_fi)
colnames(sci_fi)<-c("userId","sci_fi_down","sci_fi_up")
sci_fi[is.na(sci_fi)]<-0
sum(is.na(sci_fi$sci_fi_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(sci_fi,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Sci_Fi==1,(sci_fi_up-sci_fi_down)/2,
                  -(sci_fi_up-sci_fi_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8613835 don't include

#Adventure
adventure<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Adventure,userId)%>%
  summarize(adventure=sum(rating-b_i-b_u-mu)/(n()+l))

adventure<-adventure%>%spread(Adventure,adventure)
colnames(adventure)<-c("userId","adventure_down","adventure_up")
adventure[is.na(adventure)]<-0
sum(is.na(adventure$adventure_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(adventure,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Adventure==1,(adventure_up-adventure_down)/2,
                  -(adventure_up-adventure_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8621858 don't include

#Drama
drama<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Drama,userId)%>%
  summarize(drama=sum(rating-b_i-b_u-mu)/(n()+l))

drama<-drama%>%spread(Drama,drama)
colnames(drama)<-c("userId","drama_down","drama_up")
drama[is.na(drama)]<-0
sum(is.na(drama$drama_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8575121 include

#Fantasy
fantasy<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Fantasy,userId)%>%
  summarize(fantasy=sum(rating-b_i-b_u-mu)/(n()+l))

fantasy<-fantasy%>%spread(Fantasy,fantasy)
colnames(fantasy)<-c("userId","fantasy_down","fantasy_up")
fantasy[is.na(fantasy)]<-0
sum(is.na(fantasy$fantasy_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(fantasy,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Fantasy==1,(fantasy_up-fantasy_down)/2,
                  -(fantasy_up-fantasy_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8604285 don't include

#War
war<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(War,userId)%>%
  summarize(war=sum(rating-b_i-b_u-mu)/(n()+l))

war<-war%>%spread(War,war)
colnames(war)<-c("userId","war_down","war_up")
war[is.na(war)]<-0
sum(is.na(war$war_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(war,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(War==1,(war_up-war_down)/2,
                  -(war_up-war_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8586151 don't include

#Western
western<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Western,userId)%>%
  summarize(western=sum(rating-b_i-b_u-mu)/(n()+l))

western<-western%>%spread(Western,western)
colnames(western)<-c("userId","western_down","western_up")
western[is.na(western)]<-0
sum(is.na(western$western_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(western,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Western==1,(western_up-western_down)/2,
                  -(western_up-western_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8583809 don't include

#Mystery
mystery<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Mystery,userId)%>%
  summarize(mystery=sum(rating-b_i-b_u-mu)/(n()+l))

mystery<-mystery%>%spread(Mystery,mystery)
colnames(mystery)<-c("userId","mystery_down","mystery_up")
mystery[is.na(mystery)]<-0
sum(is.na(mystery$mystery_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(mystery,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Mystery==1,(mystery_up-mystery_down)/2,
                  -(mystery_up-mystery_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.858388 don't include

#Musical
musical<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Musical,userId)%>%
  summarize(musical=sum(rating-b_i-b_u-mu)/(n()+l))

musical<-musical%>%spread(Musical,musical)
colnames(musical)<-c("userId","musical_down","musical_up")
musical[is.na(musical)]<-0
sum(is.na(musical$musical_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(musical,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Musical==1,(musical_up-musical_down)/2,
                  -(musical_up-musical_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8603552 don't include

#Animation
animation<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Animation,userId)%>%
  summarize(animation=sum(rating-b_i-b_u-mu)/(n()+l))

animation<-animation%>%spread(Animation,animation)
colnames(animation)<-c("userId","animation_down","animation_up")
animation[is.na(animation)]<-0
sum(is.na(animation$animation_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(animation,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Animation==1,(animation_up-animation_down)/2,
                  -(animation_up-animation_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8621092 don't include

#Documentary
documentary<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Documentary,userId)%>%
  summarize(documentary=sum(rating-b_i-b_u-mu)/(n()+l))

documentary<-documentary%>%spread(Documentary,documentary)
colnames(documentary)<-c("userId","documentary_down","documentary_up")
documentary[is.na(documentary)]<-0
sum(is.na(documentary$documentary_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(documentary,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Documentary==1,(documentary_up-documentary_down)/2,
                  -(documentary_up-documentary_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8644923 don't include

#Film_noir
film_noir<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Film_noir,userId)%>%
  summarize(film_noir=sum(rating-b_i-b_u-mu)/(n()+l))

film_noir<-film_noir%>%spread(Film_noir,film_noir)
colnames(film_noir)<-c("userId","film_noir_down","film_noir_up")
film_noir[is.na(film_noir)]<-0
sum(is.na(film_noir$film_noir_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(film_noir,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Film_noir==1,(film_noir_up-film_noir_down)/2,
                  -(film_noir_up-film_noir_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8579738 don't include

#IMAX
imax<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(IMAX,userId)%>%
  summarize(imax=sum(rating-b_i-b_u-mu)/(n()+l))

imax<-imax%>%spread(IMAX,imax)
colnames(imax)<-c("userId","imax_down","imax_up")
imax[is.na(imax)]<-0
sum(is.na(imax$imax_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(imax,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(IMAX==1,(imax_up-imax_down)/2,
                  -(imax_up-imax_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8577772 don't include

#Horror
horror<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Horror,userId)%>%
  summarize(horror=sum(rating-b_i-b_u-mu)/(n()+l))

horror<-horror%>%spread(Horror,horror)
colnames(horror)<-c("userId","horror_down","horror_up")
horror[is.na(horror)]<-0
sum(is.na(horror$horror_up))

predicted_ratings <- 
  edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(crime, by= "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(horror,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)+
           ifelse(Horror==1,(horror_up-horror_down)/2,
                  -(horror_up-horror_down)/2)) %>%
  .$pred

RMSE(predicted_ratings, edx_test$rating)
#0.8577178 don't include

#Final genres Children, Comedy, Action, Crime, Drama
#
#Now optimise regularisation of genre effects
#

lambdas <- seq(40, 80, 5)

rmses <- sapply(lambdas, function(l){
  #Children
  children<-edx_train%>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId")%>%
    group_by(Children,userId)%>%
    summarize(children=sum(rating-b_i-b_u-mu)/(n()+l))
  
  children<-children%>%spread(Children,children)
  colnames(children)<-c("userId","children_down","children_up")
  children[is.na(children)]<-0
  
  #Comedy
  comedy<-edx_train%>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId")%>%
    group_by(Comedy,userId)%>%
    summarize(comedy=sum(rating-b_i-b_u-mu)/(n()+l))
  
  comedy<-comedy%>%spread(Comedy,comedy)
  colnames(comedy)<-c("userId","comedy_down","comedy_up")
  comedy[is.na(comedy)]<-0
  
  #Action
  action<-edx_train%>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId")%>%
    group_by(Action,userId)%>%
    summarize(action=sum(rating-b_i-b_u-mu)/(n()+l))
  
  action<-action%>%spread(Action,action)
  colnames(action)<-c("userId","action_down","action_up")
  action[is.na(action)]<-0
  
  #Drama
  drama<-edx_train%>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId")%>%
    group_by(Drama,userId)%>%
    summarize(drama=sum(rating-b_i-b_u-mu)/(n()+l))
  
  drama<-drama%>%spread(Drama,drama)
  colnames(drama)<-c("userId","drama_down","drama_up")
  drama[is.na(drama)]<-0
  
  #Crime
  crime<-edx_train%>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId")%>%
    group_by(Crime,userId)%>%
    summarize(crime=sum(rating-b_i-b_u-mu)/(n()+l))
  
  crime<-crime%>%spread(Crime,crime)
  colnames(crime)<-c("userId","crime_down","crime_up")
  crime[is.na(crime)]<-0
  
  predicted_ratings <- 
    edx_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(children,by = "userId")%>%
    left_join(comedy,by = "userId")%>%
    left_join(action,by = "userId")%>%
    left_join(drama,by = "userId")%>%
    left_join(crime,by = "userId")%>%
    mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                          -(children_up-children_down)/2)+
             ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                    -(comedy_up-comedy_down)/2)+
             ifelse(Action==1,(action_up-action_down)/2,
                    -(action_up-action_down)/2)+
             ifelse(Drama==1,(drama_up-drama_down)/2,
                    -(drama_up-drama_down)/2)+
             ifelse(Crime==1,(crime_up-crime_down)/2,
                    -(crime_up-crime_down)/2)) %>%
    .$pred
  return(RMSE(predicted_ratings, edx_test$rating))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

l<-lambdas[which.min(rmses)]

#Children train on optimised lambda
children<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Children,userId)%>%
  summarize(children=sum(rating-b_i-b_u-mu)/(n()+l))

children<-children%>%spread(Children,children)
colnames(children)<-c("userId","children_down","children_up")
children[is.na(children)]<-0

#Comedy
comedy<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Comedy,userId)%>%
  summarize(comedy=sum(rating-b_i-b_u-mu)/(n()+l))

comedy<-comedy%>%spread(Comedy,comedy)
colnames(comedy)<-c("userId","comedy_down","comedy_up")
comedy[is.na(comedy)]<-0


#Action
action<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Action,userId)%>%
  summarize(action=sum(rating-b_i-b_u-mu)/(n()+l))

action<-action%>%spread(Action,action)
colnames(action)<-c("userId","action_down","action_up")
action[is.na(action)]<-0


#Drama
drama<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Drama,userId)%>%
  summarize(drama=sum(rating-b_i-b_u-mu)/(n()+l))

drama<-drama%>%spread(Drama,drama)
colnames(drama)<-c("userId","drama_down","drama_up")
drama[is.na(drama)]<-0

#Crime
crime<-edx_train%>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId")%>%
  group_by(Crime,userId)%>%
  summarize(crime=sum(rating-b_i-b_u-mu)/(n()+l))

crime<-crime%>%spread(Crime,crime)
colnames(crime)<-c("userId","crime_down","crime_up")
crime[is.na(crime)]<-0

#Confirm validation data contains movies and users in edx_train

validation <- validation %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

#Add genre columns
validation_genre<-validation%>%mutate(Children=ifelse(str_detect(genres,"Children")==TRUE,1,0),
                        Comedy=ifelse(str_detect(genres,"Comedy")==TRUE,1,0),
                        Action=ifelse(str_detect(genres,"Action")==TRUE,1,0),
                        Drama=ifelse(str_detect(genres,"Drama")==TRUE,1,0),
                        Crime=ifelse(str_detect(genres,"Crime")==TRUE,1,0))%>%
  select(-genres,-title)

predicted_ratings <- validation_genre %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(children,by = "userId")%>%
  left_join(comedy,by = "userId")%>%
  left_join(action,by = "userId")%>%
  left_join(drama,by = "userId")%>%
  left_join(crime,by = "userId")%>%
  mutate(pred = mu + b_i + b_u + ifelse(Children==1,(children_up-children_down)/2,
                                        -(children_up-children_down)/2)+
           ifelse(Comedy==1,(comedy_up-comedy_down)/2,
                  -(comedy_up-comedy_down)/2)+
           ifelse(Action==1,(action_up-action_down)/2,
                  -(action_up-action_down)/2)+
           ifelse(Drama==1,(drama_up-drama_down)/2,
                  -(drama_up-drama_down)/2)+
           ifelse(Crime==1,(crime_up-crime_down)/2,
                  -(crime_up-crime_down)/2)) %>%
  .$pred
RMSE(predicted_ratings, validation$rating)
#0.8552371 - 25 points

