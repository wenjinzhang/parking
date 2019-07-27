# import the pygame module, so you can use it
import pygame
 
# define a main function
def main():

    pygame.init()

    # testImages/Sunny/2012-09-11/
    lot = pygame.image.load("/Users/Jason/Documents/GitHub/parking/testImages/Sunny/2012-09-11/2012-09-11_15_16_58.jpg")

    # pygame.display.set_icon(logo)
    pygame.display.set_caption("Parking Lot User Interface")
     
    screen = pygame.display.set_mode((1000,1000))

    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        screen.blit(lot,(50,50))

        pygame.display.update()

        
     
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()