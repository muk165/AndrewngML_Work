library(ggplot2)
df <- data.frame(dose=c("D0.5", "D1", "D2"),
                 len=c(4.2, 10, 29.5))

# Basic barplot
p<-ggplot(data=df, aes(x=dose, y=len)) +
  geom_bar(stat="identity")
p
ggsave(p, file="tooth_growth.png" , width=4, height=4)

# Horizontal bar plot
a <- p + coord_flip()

ggsave(a,file = "tooth_growth_flipped.png",width=4,height=4)


mt_cars <- ggplot(mtcars, aes(x=mpg, y=wt))+geom_point(shape=19)
ggsave(mt_cars,file = "mtcars.png",width=4,height=4)
